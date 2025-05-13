# main.py
import time
import signal
import sys
from app.controller import AgroBotController

# Create controller instance
controller = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down AgroBot...")
    if controller:
        controller.stop()
        controller.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting AgroBot...")
    
    try:
        # Initialize controller
        controller = AgroBotController()
        
        # Start the bot
        controller.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup on exit
        if controller:
            controller.stop()
            controller.cleanup()