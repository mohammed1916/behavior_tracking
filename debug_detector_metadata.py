#!/usr/bin/env python3
"""Debug script to check if detector_metadata exists in database"""

import sqlite3
import json
from backend.db import DB_PATH

def check_detector_metadata():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Get all analyses
        cur.execute("SELECT id, filename FROM analyses ORDER BY created_at DESC LIMIT 5")
        analyses = cur.fetchall()
        
        if not analyses:
            print("No analyses found in database")
            return
            
        print(f"\nFound {len(analyses)} recent analyses:")
        for aid, fname in analyses:
            print(f"\n  ID: {aid}, File: {fname}")
            
            # Check samples for this analysis
            cur.execute(
                "SELECT COUNT(*), COUNT(CASE WHEN detector_metadata IS NOT NULL THEN 1 END) FROM samples WHERE analysis_id=?",
                (aid,)
            )
            total, with_metadata = cur.fetchone()
            print(f"    Samples: {total} total, {with_metadata} with detector_metadata")
            
            # Show first sample with detector metadata
            if with_metadata > 0:
                cur.execute(
                    "SELECT frame_index, detector_metadata FROM samples WHERE analysis_id=? AND detector_metadata IS NOT NULL LIMIT 1",
                    (aid,)
                )
                row = cur.fetchone()
                if row:
                    frame_idx, metadata_json = row
                    print(f"    First detector sample at frame {frame_idx}:")
                    try:
                        metadata = json.loads(metadata_json)
                        for key in metadata:
                            if key != 'detections':  # Skip detailed detection data
                                val = str(metadata[key])[:100]
                                print(f"      {key}: {val}")
                            else:
                                print(f"      {key}: (contains detections)")
                    except Exception as e:
                        print(f"      ERROR parsing JSON: {e}")
                        print(f"      Raw: {metadata_json[:200]}")
            else:
                print(f"    No detector metadata found in any samples")
        
        conn.close()
        
    except FileNotFoundError:
        print(f"Database not found at {DB_PATH}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_detector_metadata()
