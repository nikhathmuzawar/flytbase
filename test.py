from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
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
from dataclasses import dataclass

# Drone system classes (embedded)
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

@dataclass
class Conflict:
    primary_drone_id: str
    conflicting_drone_id: str
    conflict_location: tuple
    conflict_time: datetime
    distance: float
    description: str

class DroneDeconflictionSystem:
    def __init__(self, safety_buffer: float = 50.0):
        self.safety_buffer = safety_buffer
        self.reference_point = None
        self.primary_drone = None
        self.simulated_drones = []
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
        
        # Check if this is QGC format
        if lines[0].strip().startswith('QGC WPL'):
            print(f"Detected QGC format for {drone_id}")
            return self._parse_qgc_format(lines, drone_id, start_time, end_time, speed)
        
        # Original CSV format detection
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
        
        print(f"Detected CSV format ({format_detected}) for {drone_id}")
        
        # Process CSV waypoints
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
        """Parse QGC waypoint format with proper altitude handling"""
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
                    print(f"  Added waypoint {seq}: ({lat:.6f}, {lon:.6f}, {relative_alt:.1f}m relative)")
                
            except (ValueError, IndexError) as e:
                print(f"  Skipping invalid line: {line[:50]}... Error: {e}")
                continue
        
        print(f"Parsed {len(waypoints)} valid waypoints from QGC format")
        print(f"Ground altitude reference: {ground_alt}m ASL")
        
        if not waypoints:
            raise ValueError("No valid waypoints found in QGC file")
        
        return DroneTrajectory(drone_id, waypoints, start_time, end_time, speed)
    
    def set_primary_mission(self, trajectory):
        self.primary_drone = trajectory
        self._interpolate_waypoint_times(trajectory)
    
    def add_simulated_drone(self, trajectory):
        self._interpolate_waypoint_times(trajectory)
        self.simulated_drones.append(trajectory)
    
    def _interpolate_waypoint_times(self, trajectory):
        if not trajectory.waypoints:
            return
        
        print(f"\n=== TIME INTERPOLATION DEBUG for {trajectory.drone_id} ===")
        print(f"Mission duration: {trajectory.start_time} to {trajectory.end_time}")
        print(f"Speed: {trajectory.speed} m/s")
        
        trajectory.waypoints[0].timestamp = trajectory.start_time
        current_time = trajectory.start_time
        total_distance = 0
        
        for i in range(1, len(trajectory.waypoints)):
            prev_wp = trajectory.waypoints[i-1]
            curr_wp = trajectory.waypoints[i]
            
            distance = np.sqrt(
                (curr_wp.x - prev_wp.x)**2 + 
                (curr_wp.y - prev_wp.y)**2 + 
                (curr_wp.z - prev_wp.z)**2
            )
            total_distance += distance
        
        print(f"Total path distance: {total_distance:.1f}m")
        
        current_time = trajectory.start_time
        
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
                
                print(f"  WP{i}: distance={distance:.1f}m, time={travel_time:.1f}s, arrives at {current_time}")
            else:
                curr_wp.timestamp = current_time
                print(f"  WP{i}: same position as previous, arrives at {current_time}")
        
        actual_end_time = trajectory.waypoints[-1].timestamp
        scheduled_end_time = trajectory.end_time
        
        print(f"Calculated end time: {actual_end_time}")
        print(f"Scheduled end time: {scheduled_end_time}")
        
        if actual_end_time > scheduled_end_time:
            print(f"WARNING: Mission extends beyond scheduled time by {(actual_end_time - scheduled_end_time).total_seconds():.1f} seconds")
            mission_duration = (scheduled_end_time - trajectory.start_time).total_seconds()
            actual_duration = (actual_end_time - trajectory.start_time).total_seconds()
            
            if actual_duration > 0:
                time_scale_factor = mission_duration / actual_duration
                print(f"Scaling time by factor: {time_scale_factor:.3f}")
                
                for i in range(1, len(trajectory.waypoints)):
                    elapsed_time = (trajectory.waypoints[i].timestamp - trajectory.start_time).total_seconds()
                    scaled_elapsed_time = elapsed_time * time_scale_factor
                    trajectory.waypoints[i].timestamp = trajectory.start_time + timedelta(seconds=scaled_elapsed_time)
        
        print(f"=== END TIME INTERPOLATION ===\n")
    
    def get_drone_position_at_time(self, trajectory, target_time):
        if target_time < trajectory.start_time or target_time > trajectory.end_time:
            return None
        
        waypoints = trajectory.waypoints
        
        if len(waypoints) == 1:
            return (waypoints[0].x, waypoints[0].y, waypoints[0].z)
        
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
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
        
        last_wp = waypoints[-1]
        return (last_wp.x, last_wp.y, last_wp.z)
    
    def check_conflicts(self):
        self.conflicts = []
        
        if not self.primary_drone:
            return self.conflicts
        
        print(f"\n=== DETAILED CONFLICT DETECTION DEBUG ===")
        
        other_start_times = [drone.start_time for drone in self.simulated_drones]
        other_end_times = [drone.end_time for drone in self.simulated_drones]
        
        earliest_common_start = max([self.primary_drone.start_time] + other_start_times)
        latest_common_end = min([self.primary_drone.end_time] + other_end_times)
        
        if earliest_common_start >= latest_common_end:
            print("No temporal overlap - missions don't fly at the same time!")
            return self.conflicts
        
        time_step = timedelta(seconds=1)
        current_time = earliest_common_start
        
        while current_time <= latest_common_end:
            primary_pos = self.get_drone_position_at_time(self.primary_drone, current_time)
            
            if primary_pos:
                for sim_drone in self.simulated_drones:
                    sim_pos = self.get_drone_position_at_time(sim_drone, current_time)
                    
                    if sim_pos:
                        distance = np.sqrt(
                            (primary_pos[0] - sim_pos[0])**2 +
                            (primary_pos[1] - sim_pos[1])**2 +
                            (primary_pos[2] - sim_pos[2])**2
                        )
                        
                        if distance < self.safety_buffer:
                            is_duplicate = False
                            for existing_conflict in self.conflicts:
                                if (existing_conflict.conflicting_drone_id == sim_drone.drone_id and
                                    abs((existing_conflict.conflict_time - current_time).total_seconds()) < 5):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                conflict = Conflict(
                                    primary_drone_id=self.primary_drone.drone_id,
                                    conflicting_drone_id=sim_drone.drone_id,
                                    conflict_location=(
                                        (primary_pos[0] + sim_pos[0]) / 2,
                                        (primary_pos[1] + sim_pos[1]) / 2,
                                        (primary_pos[2] + sim_pos[2]) / 2
                                    ),
                                    conflict_time=current_time,
                                    distance=distance,
                                    description=f"Conflict at distance {distance:.2f}m at {current_time}"
                                )
                                self.conflicts.append(conflict)
            
            current_time += time_step
        
        print(f"=== END CONFLICT DETECTION DEBUG ===\n")
        
        return self.conflicts
    
    def get_conflict_report(self):
        conflicts = self.check_conflicts()
        
        if not conflicts:
            return {
                "status": "clear",
                "message": "No conflicts detected",
                "conflicts": [],
                "detailed_summary": []
            }
        
        conflict_summary = {}
        for conflict in conflicts:
            drone_id = conflict.conflicting_drone_id
            if drone_id not in conflict_summary:
                conflict_summary[drone_id] = {
                    "count": 0,
                    "min_distance": float('inf'),
                    "max_distance": 0.0,
                    "avg_distance": 0.0,
                    "conflicts": [],
                    "first_conflict_time": None,
                    "last_conflict_time": None
                }
            
            conflict_summary[drone_id]["count"] += 1
            conflict_summary[drone_id]["min_distance"] = min(
                conflict_summary[drone_id]["min_distance"], 
                conflict.distance
            )
            conflict_summary[drone_id]["max_distance"] = max(
                conflict_summary[drone_id]["max_distance"], 
                conflict.distance
            )
            
            if conflict_summary[drone_id]["first_conflict_time"] is None:
                conflict_summary[drone_id]["first_conflict_time"] = conflict.conflict_time
            conflict_summary[drone_id]["last_conflict_time"] = conflict.conflict_time
            
            conflict_summary[drone_id]["conflicts"].append({
                "time": conflict.conflict_time.isoformat(),
                "location": conflict.conflict_location,
                "distance": conflict.distance,
                "description": conflict.description
            })
        
        # Calculate average distances
        for drone_id, summary in conflict_summary.items():
            total_dist = sum(c["distance"] for c in summary["conflicts"])
            summary["avg_distance"] = total_dist / summary["count"]
        
        # Create detailed summary list
        detailed_summary = []
        for drone_id, summary in conflict_summary.items():
            detailed_summary.append({
                "drone_id": drone_id,
                "total_conflicts": summary["count"],
                "min_distance": summary["min_distance"],
                "max_distance": summary["max_distance"],
                "avg_distance": summary["avg_distance"],
                "first_conflict": summary["first_conflict_time"].isoformat() if summary["first_conflict_time"] else None,
                "last_conflict": summary["last_conflict_time"].isoformat() if summary["last_conflict_time"] else None,
                "duration_seconds": (summary["last_conflict_time"] - summary["first_conflict_time"]).total_seconds() if summary["first_conflict_time"] and summary["last_conflict_time"] else 0
            })
        
        return {
            "status": "conflict_detected",
            "total_conflicts": len(conflicts),
            "conflicting_drones": len(conflict_summary),
            "summary": conflict_summary,
            "detailed_summary": detailed_summary
        }

# FastAPI app
app = FastAPI(title="Drone Deconfliction System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("test.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

@app.post("/analyze")
async def analyze_drone_paths(
    primary_file: UploadFile = File(...),
    primary_start_time: str = Form(...),
    primary_end_time: str = Form(...),
    primary_speed: float = Form(10.0),
    other_files: List[UploadFile] = File(...),
    other_start_times: List[str] = Form(...),
    other_end_times: List[str] = Form(...),
    other_speeds: List[str] = Form(...),
    other_drone_ids: List[str] = Form(...),
    safety_buffer: float = Form(60.0)
):
    try:
        print(f"Received request: safety_buffer={safety_buffer}")
        
        if len(other_files) < 1 or len(other_files) > 10:
            raise HTTPException(status_code=400, detail="Must upload 1-10 other drone waypoint files")
        
        if not (len(other_files) == len(other_start_times) == len(other_end_times) == 
                len(other_speeds) == len(other_drone_ids)):
            raise HTTPException(status_code=400, detail="Mismatch in other drone data arrays")
        
        system = DroneDeconflictionSystem(safety_buffer=safety_buffer)
        
        def parse_datetime(dt_str):
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        primary_start = parse_datetime(primary_start_time)
        primary_end = parse_datetime(primary_end_time)
        
        try:
            primary_content = (await primary_file.read()).decode('utf-8')
            primary_trajectory = system.load_waypoint_content(
                primary_content, "primary", primary_start, primary_end, primary_speed
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing primary file: {str(e)}")
        
        system.set_primary_mission(primary_trajectory)
        
        for i, (file, start_time_str, end_time_str, speed_str, drone_id) in enumerate(
            zip(other_files, other_start_times, other_end_times, other_speeds, other_drone_ids)
        ):
            try:
                start_time = parse_datetime(start_time_str)
                end_time = parse_datetime(end_time_str)
                speed = float(speed_str)
                
                content = (await file.read()).decode('utf-8')
                trajectory = system.load_waypoint_content(content, drone_id, start_time, end_time, speed)
                system.add_simulated_drone(trajectory)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing file {file.filename}: {str(e)}")
        
        try:
            conflict_report = system.get_conflict_report()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing conflicts: {str(e)}")
        
        try:
            plot_data = generate_plots(system)
        except Exception as e:
            plot_data = {
                "plot_2d": "data:image/png;base64,",
                "plot_3d": "data:image/png;base64,"
            }
        
        earliest_time = min(primary_start, min(parse_datetime(t) for t in other_start_times))
        latest_time = max(primary_end, max(parse_datetime(t) for t in other_end_times))
        analysis_duration = f"{int((latest_time - earliest_time).total_seconds() / 60)}min"
        
        return {
            "status": "success",
            "conflict_report": conflict_report,
            "plots": plot_data,
            "system_info": {
                "safety_buffer": safety_buffer,
                "analysis_duration": analysis_duration,
                "total_drones": len(other_files) + 1
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def generate_plots(system):
    # Generate single 2D plot with local coordinates only
    fig_2d, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Local coordinates
    if system.primary_drone:
        x_coords = [wp.x for wp in system.primary_drone.waypoints]
        y_coords = [wp.y for wp in system.primary_drone.waypoints]
        ax.plot(x_coords, y_coords, 'b-o', linewidth=3, markersize=8, label='Primary Drone', alpha=0.8)
        ax.scatter(x_coords[0], y_coords[0], c='blue', s=150, marker='s', label='Primary Start', alpha=0.9)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, drone in enumerate(system.simulated_drones):
        color = colors[i % len(colors)]
        x_coords = [wp.x for wp in drone.waypoints]
        y_coords = [wp.y for wp in drone.waypoints]
        ax.plot(x_coords, y_coords, f'{color[0]}-s', linewidth=2, markersize=6, 
                label=drone.drone_id, alpha=0.7)
        ax.scatter(x_coords[0], y_coords[0], c=color, s=100, marker='^', alpha=0.9)
    
    # Plot conflicts
    for conflict in system.conflicts:
        ax.scatter(conflict.conflict_location[0], conflict.conflict_location[1], 
                   c='red', s=200, marker='X', alpha=0.9, edgecolors='darkred', linewidths=2)
        circle = plt.Circle((conflict.conflict_location[0], conflict.conflict_location[1]), 
                           system.safety_buffer, fill=False, color='red', alpha=0.3, linestyle='--', linewidth=2)
        ax.add_patch(circle)
    
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title('2D Trajectory View - Local Coordinates', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer_2d = BytesIO()
    plt.savefig(buffer_2d, format='png', dpi=150, bbox_inches='tight')
    buffer_2d.seek(0)
    plot_2d_base64 = base64.b64encode(buffer_2d.getvalue()).decode()
    plt.close()
    
    # Generate 3D plot
    fig_3d = plt.figure(figsize=(12, 8))
    ax = fig_3d.add_subplot(111, projection='3d')
    
    if system.primary_drone:
        x_coords = [wp.x for wp in system.primary_drone.waypoints]
        y_coords = [wp.y for wp in system.primary_drone.waypoints]
        z_coords = [wp.z for wp in system.primary_drone.waypoints]
        ax.plot(x_coords, y_coords, z_coords, 'b-o', linewidth=3, markersize=8, 
               label='Primary Drone', alpha=0.8)
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], c='blue', s=150, marker='s', alpha=0.9)
    
    for i, drone in enumerate(system.simulated_drones):
        color = colors[i % len(colors)]
        x_coords = [wp.x for wp in drone.waypoints]
        y_coords = [wp.y for wp in drone.waypoints]
        z_coords = [wp.z for wp in drone.waypoints]
        ax.plot(x_coords, y_coords, z_coords, f'{color[0]}-s', linewidth=2, markersize=6, 
               label=drone.drone_id, alpha=0.7)
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], c=color, s=100, marker='^', alpha=0.9)
    
    for conflict in system.conflicts:
        ax.scatter(conflict.conflict_location[0], conflict.conflict_location[1], 
                  conflict.conflict_location[2], c='red', s=300, marker='X', alpha=0.9,
                  edgecolors='darkred', linewidths=2)
    
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_zlabel('Altitude (m)', fontsize=12)
    ax.set_title('3D Trajectories and Conflicts', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)