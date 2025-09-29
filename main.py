from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
from io import BytesIO
import numpy as np
import math
from dataclasses import dataclass, asdict
from enum import Enum

# Mission approval status
class MissionStatus(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"

@dataclass
class Waypoint:
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.alt != 0.0:
            self.z = self.alt

@dataclass
class DroneTrajectory:
    drone_id: str
    waypoints: List[Waypoint]
    start_time: datetime
    end_time: datetime
    speed: float = 10.0
    status: MissionStatus = MissionStatus.PENDING

@dataclass
class Conflict:
    primary_drone_id: str
    conflicting_drone_id: str
    conflict_location: tuple
    conflict_time: datetime
    distance: float
    description: str

@dataclass
class DeconflictionSuggestion:
    suggestion_type: str
    description: str
    parameters: Dict

class DroneDeconflictionService:
    def __init__(self, safety_buffer: float = 50.0):
        self.safety_buffer = safety_buffer
        self.reference_point = None
        self.approved_missions: Dict[str, DroneTrajectory] = {}
        self.conflicts = []
    
    def lat_lon_to_local(self, lat: float, lon: float):
        if self.reference_point is None:
            self.reference_point = (lat, lon)
            return (0.0, 0.0)
        
        ref_lat, ref_lon = self.reference_point
        R = 6371000.0
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        dlat = lat_rad - ref_lat_rad
        dlon = lon_rad - ref_lon_rad
        
        x = R * dlon * math.cos(ref_lat_rad)
        y = R * dlat
        
        return (x, y)
    
    def local_to_lat_lon(self, x: float, y: float):
        if self.reference_point is None:
            return (0.0, 0.0)
        
        ref_lat, ref_lon = self.reference_point
        R = 6371000.0
        
        ref_lat_rad = math.radians(ref_lat)
        dlat = y / R
        dlon = x / (R * math.cos(ref_lat_rad))
        
        lat = ref_lat + math.degrees(dlat)
        lon = ref_lon + math.degrees(dlon)
        
        return (lat, lon)
    
    def load_waypoint_content(self, content: str, drone_id: str, start_time: datetime, end_time: datetime, speed: float):
        waypoints = []
        lines = content.strip().split('\n')
        
        if lines[0].strip().startswith('QGC WPL'):
            return self._parse_qgc_format(lines, drone_id, start_time, end_time, speed)
        
        format_detected = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                coords = [float(x) for x in line.split(',')]
                if len(coords) >= 2:
                    if -90 <= coords[0] <= 90 and -180 <= coords[1] <= 180:
                        format_detected = 'gps'
                    else:
                        format_detected = 'cartesian'
                break
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                coords = [float(x) for x in line.split(',')]
                
                if format_detected == 'gps':
                    lat = coords[0]
                    lon = coords[1]
                    alt = coords[2] if len(coords) > 2 else 0.0
                    
                    x, y = self.lat_lon_to_local(lat, lon)
                    waypoint = Waypoint(lat=lat, lon=lon, alt=alt, x=x, y=y, z=alt)
                    waypoints.append(waypoint)
                else:
                    x = coords[0]
                    y = coords[1]
                    z = coords[2] if len(coords) > 2 else 0.0
                    
                    lat, lon = self.local_to_lat_lon(x, y)
                    waypoint = Waypoint(lat=lat, lon=lon, alt=z, x=x, y=y, z=z)
                    waypoints.append(waypoint)
        
        return DroneTrajectory(drone_id, waypoints, start_time, end_time, speed)
    
    def _parse_qgc_format(self, lines, drone_id, start_time, end_time, speed):
        waypoints = []
        ground_alt = None
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 11:
                continue
            
            try:
                seq = int(parts[0])
                command = int(parts[3])
                lat = float(parts[8])
                lon = float(parts[9])
                alt = float(parts[10])
                
                if command == 16 or (command == 22 and lat != 0 and lon != 0):
                    if lat == 0 and lon == 0:
                        continue
                    
                    if ground_alt is None:
                        ground_alt = alt
                        relative_alt = 0.0
                    else:
                        relative_alt = alt - ground_alt
                        relative_alt = max(0.0, relative_alt)
                    
                    x, y = self.lat_lon_to_local(lat, lon)
                    waypoint = Waypoint(lat=lat, lon=lon, alt=relative_alt, x=x, y=y, z=relative_alt)
                    waypoints.append(waypoint)
                
            except (ValueError, IndexError):
                continue
        
        if not waypoints:
            raise ValueError("No valid waypoints found in QGC file")
        
        return DroneTrajectory(drone_id, waypoints, start_time, end_time, speed)
    
    def _interpolate_waypoint_times(self, trajectory):
        """Interpolate timestamps for waypoints based on trajectory timing and speed"""
        if not trajectory.waypoints:
            return
        
        # Ensure all waypoints have timestamps initialized
        for wp in trajectory.waypoints:
            if wp.timestamp is None:
                wp.timestamp = trajectory.start_time
        
        # Set first waypoint timestamp
        trajectory.waypoints[0].timestamp = trajectory.start_time
        current_time = trajectory.start_time
        
        # Calculate timestamps for intermediate waypoints based on distance and speed
        for i in range(1, len(trajectory.waypoints)):
            prev_wp = trajectory.waypoints[i-1]
            curr_wp = trajectory.waypoints[i]
            
            distance = np.sqrt(
                (curr_wp.x - prev_wp.x)**2 + 
                (curr_wp.y - prev_wp.y)**2 + 
                (curr_wp.z - prev_wp.z)**2
            )
            
            if distance > 0:
                travel_time = distance / trajectory.speed
                current_time += timedelta(seconds=travel_time)
            
            curr_wp.timestamp = current_time
        
        # Scale timing if mission exceeds scheduled end time
        actual_end_time = trajectory.waypoints[-1].timestamp
        scheduled_end_time = trajectory.end_time
        
        if actual_end_time > scheduled_end_time:
            mission_duration = (scheduled_end_time - trajectory.start_time).total_seconds()
            actual_duration = (actual_end_time - trajectory.start_time).total_seconds()
            
            if actual_duration > 0:
                time_scale_factor = mission_duration / actual_duration
                
                # Rescale all waypoint timestamps
                for i in range(1, len(trajectory.waypoints)):
                    elapsed_time = (trajectory.waypoints[i].timestamp - trajectory.start_time).total_seconds()
                    scaled_elapsed_time = elapsed_time * time_scale_factor
                    trajectory.waypoints[i].timestamp = trajectory.start_time + timedelta(seconds=scaled_elapsed_time)
    
    def get_drone_position_at_time(self, trajectory, target_time):
        """Get interpolated drone position at a specific time"""
        if target_time < trajectory.start_time or target_time > trajectory.end_time:
            return None
        
        waypoints = trajectory.waypoints
        
        # Ensure waypoints have timestamps
        if not waypoints[0].timestamp:
            self._interpolate_waypoint_times(trajectory)
        
        if len(waypoints) == 1:
            return (waypoints[0].x, waypoints[0].y, waypoints[0].z)
        
        # Find the segment containing the target time
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            # Ensure timestamps are not None
            if wp1.timestamp is None or wp2.timestamp is None:
                continue
                
            if wp1.timestamp <= target_time <= wp2.timestamp:
                total_duration = (wp2.timestamp - wp1.timestamp).total_seconds()
                if total_duration == 0:
                    return (wp1.x, wp1.y, wp1.z)
                
                elapsed = (target_time - wp1.timestamp).total_seconds()
                ratio = elapsed / total_duration
                
                x = wp1.x + (wp2.x - wp1.x) * ratio
                y = wp1.y + (wp2.y - wp1.y) * ratio
                z = wp1.z + (wp2.z - wp1.z) * ratio
                
                return (x, y, z)
        
        # Return last waypoint position if not found in segments
        last_wp = waypoints[-1]
        return (last_wp.x, last_wp.y, last_wp.z)
    
    def check_conflicts_against_approved(self, proposed_mission: DroneTrajectory):
        """Check for conflicts between proposed mission and all approved missions"""
        self.conflicts = []
        
        if not self.approved_missions:
            return self.conflicts
        
        # Ensure proposed mission has interpolated timestamps
        self._interpolate_waypoint_times(proposed_mission)
        
        for approved_id, approved_mission in self.approved_missions.items():
            # Ensure approved mission has interpolated timestamps
            if not approved_mission.waypoints[0].timestamp:
                self._interpolate_waypoint_times(approved_mission)
            
            # Find overlapping time window
            earliest_common_start = max(proposed_mission.start_time, approved_mission.start_time)
            latest_common_end = min(proposed_mission.end_time, approved_mission.end_time)
            
            if earliest_common_start >= latest_common_end:
                continue  # No temporal overlap
            
            # Check for conflicts at regular time intervals
            time_step = timedelta(seconds=1)
            current_time = earliest_common_start
            
            while current_time <= latest_common_end:
                proposed_pos = self.get_drone_position_at_time(proposed_mission, current_time)
                approved_pos = self.get_drone_position_at_time(approved_mission, current_time)
                
                if proposed_pos and approved_pos:
                    distance = np.sqrt(
                        (proposed_pos[0] - approved_pos[0])**2 +
                        (proposed_pos[1] - approved_pos[1])**2 +
                        (proposed_pos[2] - approved_pos[2])**2
                    )
                    
                    if distance < self.safety_buffer:
                        # Check for duplicate conflicts (within 5 seconds)
                        is_duplicate = False
                        for existing_conflict in self.conflicts:
                            if (existing_conflict.conflicting_drone_id == approved_id and
                                abs((existing_conflict.conflict_time - current_time).total_seconds()) < 5):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            conflict = Conflict(
                                primary_drone_id=proposed_mission.drone_id,
                                conflicting_drone_id=approved_id,
                                conflict_location=(
                                    (proposed_pos[0] + approved_pos[0]) / 2,
                                    (proposed_pos[1] + approved_pos[1]) / 2,
                                    (proposed_pos[2] + approved_pos[2]) / 2
                                ),
                                conflict_time=current_time,
                                distance=distance,
                                description=f"Conflict with {approved_id} at distance {distance:.2f}m"
                            )
                            self.conflicts.append(conflict)
                
                current_time += time_step
        
        return self.conflicts
    
    def generate_deconfliction_suggestions(self, conflicts):
        """Generate suggestions to resolve conflicts"""
        if not conflicts:
            return []
        
        suggestions = []
        
        # Time shift suggestion
        conflict_times = [c.conflict_time for c in conflicts]
        earliest_conflict = min(conflict_times)
        latest_conflict = max(conflict_times)
        
        time_shift_minutes = 5
        suggestions.append(DeconflictionSuggestion(
            suggestion_type="time_delay",
            description=f"Delay mission start by {time_shift_minutes} minutes",
            parameters={
                "delay_minutes": time_shift_minutes,
                "reason": "Avoids temporal overlap with conflicting missions"
            }
        ))
        
        # Altitude adjustment
        min_distance = min(c.distance for c in conflicts)
        altitude_increase = math.ceil(self.safety_buffer - min_distance + 10)
        suggestions.append(DeconflictionSuggestion(
            suggestion_type="altitude_change",
            description=f"Increase mission altitude by {altitude_increase}m",
            parameters={
                "altitude_offset": altitude_increase,
                "reason": "Creates vertical separation from conflicting traffic"
            }
        ))
        
        # Speed adjustment
        suggestions.append(DeconflictionSuggestion(
            suggestion_type="speed_change",
            description="Reduce speed by 20% to desynchronize with conflicting drones",
            parameters={
                "speed_multiplier": 0.8,
                "reason": "Changes temporal profile to avoid conflicts"
            }
        ))
        
        return suggestions
    
    def submit_mission(self, mission: DroneTrajectory):
        """Submit a mission for approval"""
        conflicts = self.check_conflicts_against_approved(mission)
        
        if not conflicts:
            mission.status = MissionStatus.APPROVED
            self.approved_missions[mission.drone_id] = mission
            return {
                "status": MissionStatus.APPROVED.value,
                "mission_id": mission.drone_id,
                "message": "Mission approved - no conflicts detected",
                "conflicts": [],
                "suggestions": []
            }
        else:
            mission.status = MissionStatus.REJECTED
            suggestions = self.generate_deconfliction_suggestions(conflicts)
            
            # Create conflict summary
            conflict_summary = {}
            for conflict in conflicts:
                drone_id = conflict.conflicting_drone_id
                if drone_id not in conflict_summary:
                    conflict_summary[drone_id] = {
                        "count": 0,
                        "min_distance": float('inf'),
                        "conflicts": []
                    }
                
                conflict_summary[drone_id]["count"] += 1
                conflict_summary[drone_id]["min_distance"] = min(
                    conflict_summary[drone_id]["min_distance"], 
                    conflict.distance
                )
                conflict_summary[drone_id]["conflicts"].append({
                    "time": conflict.conflict_time.isoformat(),
                    "location": conflict.conflict_location,
                    "distance": conflict.distance
                })
            
            return {
                "status": MissionStatus.REJECTED.value,
                "mission_id": mission.drone_id,
                "message": f"Mission rejected - {len(conflicts)} conflicts detected with {len(conflict_summary)} drone(s)",
                "conflicts": conflict_summary,
                "suggestions": [
                    {
                        "type": s.suggestion_type,
                        "description": s.description,
                        "parameters": s.parameters
                    } for s in suggestions
                ]
            }
    
    def get_approved_missions(self):
        """Get list of approved mission IDs"""
        return list(self.approved_missions.keys())
    
    def clear_approved_missions(self):
        """Clear all approved missions"""
        self.approved_missions.clear()
        self.conflicts = []

# Global service instance
deconfliction_service = DroneDeconflictionService(safety_buffer=60.0)

# FastAPI app
app = FastAPI(title="Drone Deconfliction Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

@app.post("/api/submit_mission")
async def submit_mission(
    drone_id: str = Form(...),
    waypoint_file: UploadFile = File(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    speed: float = Form(10.0)
):
    """
    Submit a mission for deconfliction approval.
    Returns approval/rejection with conflict details and suggestions.
    """
    try:
        def parse_datetime(dt_str):
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        mission_start = parse_datetime(start_time)
        mission_end = parse_datetime(end_time)
        
        content = (await waypoint_file.read()).decode('utf-8')
        trajectory = deconfliction_service.load_waypoint_content(
            content, drone_id, mission_start, mission_end, speed
        )
        
        result = deconfliction_service.submit_mission(trajectory)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing mission: {str(e)}")

@app.get("/api/approved_missions")
async def get_approved_missions():
    """Get list of all approved mission IDs"""
    return {
        "approved_missions": deconfliction_service.get_approved_missions(),
        "count": len(deconfliction_service.approved_missions)
    }

@app.post("/api/clear_missions")
async def clear_missions():
    """Clear all approved missions (for testing/reset)"""
    deconfliction_service.clear_approved_missions()
    return {"message": "All missions cleared", "approved_missions": []}

@app.post("/api/visualize")
async def visualize_airspace():
    """Generate visualization of approved missions"""
    try:
        plot_data = generate_plots(deconfliction_service)
        return {"plots": plot_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

def generate_plots(service):
    """Generate 2D and 3D visualizations of approved missions"""
    if not service.approved_missions:
        return {
            "plot_2d": "data:image/png;base64,",
            "plot_3d": "data:image/png;base64,"
        }
    
    # 2D plot
    fig_2d, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (drone_id, mission) in enumerate(service.approved_missions.items()):
        color = colors[i % len(colors)]
        x_coords = [wp.x for wp in mission.waypoints]
        y_coords = [wp.y for wp in mission.waypoints]
        ax.plot(x_coords, y_coords, f'{color[0]}-o', linewidth=2, markersize=6, 
                label=drone_id, alpha=0.7)
        ax.scatter(x_coords[0], y_coords[0], c=color, s=150, marker='s', alpha=0.9)
    
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title('Approved Missions - 2D View', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    buffer_2d = BytesIO()
    plt.savefig(buffer_2d, format='png', dpi=150, bbox_inches='tight')
    buffer_2d.seek(0)
    plot_2d_base64 = base64.b64encode(buffer_2d.getvalue()).decode()
    plt.close()
    
    # 3D plot
    fig_3d = plt.figure(figsize=(12, 8))
    ax = fig_3d.add_subplot(111, projection='3d')
    
    for i, (drone_id, mission) in enumerate(service.approved_missions.items()):
        color = colors[i % len(colors)]
        x_coords = [wp.x for wp in mission.waypoints]
        y_coords = [wp.y for wp in mission.waypoints]
        z_coords = [wp.z for wp in mission.waypoints]
        ax.plot(x_coords, y_coords, z_coords, f'{color[0]}-o', linewidth=2, markersize=6, 
               label=drone_id, alpha=0.7)
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], c=color, s=150, marker='s', alpha=0.9)
    
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_zlabel('Altitude (m)', fontsize=12)
    ax.set_title('Approved Missions - 3D View', fontsize=14, fontweight='bold')
    ax.legend()
    
    buffer_3d = BytesIO()
    plt.savefig(buffer_3d, format='png', dpi=150, bbox_inches='tight')
    buffer_3d.seek(0)
    plot_3d_base64 = base64.b64encode(buffer_3d.getvalue()).decode()
    plt.close()
    
    return {
        "plot_2d": f"data:image/png;base64,{plot_2d_base64}",
        "plot_3d": f"data:image/png;base64,{plot_3d_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)