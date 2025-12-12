import requests
import sys

def analyze_video(video_path, server_url="http://localhost:8000"):
    url = f"{server_url}/analyze_video"
    files = {'file': open(video_path, 'rb')}
    
    print(f"Uploading {video_path} to {url}...")
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        result = response.json()
        print("Analysis Complete!")
        print(f"Task Completed: {result.get('task_completed')}")
        print(f"Download URL: {result.get('download_url')}")
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_video>")
    else:
        analyze_video(sys.argv[1])
