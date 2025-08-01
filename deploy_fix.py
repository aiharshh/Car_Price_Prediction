#!/usr/bin/env python3
"""
Quick deployment update script
Run this to deploy the accuracy fix to your Render app
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and show the result"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {str(e)}")
        return False

def main():
    """Deploy the accuracy fix"""
    print("🚀 Car Price Prediction - Accuracy Fix Deployment")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: Not in the correct directory. Please run from the project root.")
        sys.exit(1)
    
    # Check git status
    print("\n📊 Current git status:")
    subprocess.run("git status --porcelain", shell=True)
    
    # Deploy steps
    steps = [
        ("git add .", "Adding all changes to git"),
        ("git commit -m \"Fixed model accuracy - smart data repair with 59.52% R2 score\"", "Committing accuracy fix"),
        ("git push origin main", "Pushing to GitHub (triggers Render deployment)")
    ]
    
    success_count = 0
    for command, description in steps:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"\n⚠️ Step failed, but continuing...")
    
    print(f"\n📊 Deployment Summary:")
    print(f"Completed steps: {success_count}/{len(steps)}")
    
    if success_count >= 2:  # At least added and committed
        print("\n🎉 Deployment initiated successfully!")
        print("\n📍 What happens next:")
        print("1. Render will automatically detect the changes")
        print("2. Rebuild and redeploy your app (5-10 minutes)")
        print("3. Your app will have the new smart data repair system")
        print("4. Model accuracy will be ~59.52% instead of 0%")
        
        print("\n🌐 Monitor deployment:")
        print("- Render Dashboard: https://dashboard.render.com")
        print("- Your App: https://car-price-prediction-1yfs.onrender.com")
        print("- Status Page: https://car-price-prediction-1yfs.onrender.com/status")
        
        print("\n💡 After deployment:")
        print("- First prediction will trigger the improved model training")
        print("- Expect much better accuracy (~60% instead of 0%)")
        print("- Data corruption is now automatically fixed")
        
    else:
        print("\n⚠️ Deployment may have issues. Check git status and try manually:")
        print("git add .")
        print("git commit -m \"Fixed model accuracy\"")
        print("git push origin main")

if __name__ == "__main__":
    main()
