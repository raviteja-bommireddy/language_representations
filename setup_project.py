#!/usr/bin/env python3
"""
Setup script for Language Representations project
This script helps set up the environment and download necessary resources
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please install manually.")
        return False
    return True

def download_evaluation_datasets():
    """Download evaluation datasets"""
    print("📊 Downloading evaluation datasets...")
    
    # SimLex-999
    try:
        urllib.request.urlretrieve(
            'https://fh295.github.io/SimLex-999.txt',
            'SimLex-999.txt'
        )
        print("✅ Downloaded SimLex-999")
    except Exception as e:
        print(f"⚠️  Could not download SimLex-999: {e}")
        # Create sample data
        sample_simlex = """word1\tword2\tPOS\tSimLex999\tconc(w1)\tconc(w2)\tUSF(w1,w2)\tUSF(w2,w1)\tSD(SimLex)
old\tnew\tA\t1.58\t2.72\t2.81\t7.25\t5.94\t0.41
smart\tintelligent\tA\t9.2\t1.75\t2.46\t7.11\t6.85\t0.67
hard\tdifficult\tA\t8.77\t3.76\t1.19\t5.94\t4.94\t0.95
happy\tcheerful\tA\t9.55\t2.56\t2.34\t5.85\t4.24\t0.95"""
        with open('SimLex-999.txt', 'w') as f:
            f.write(sample_simlex)
        print("✅ Created sample SimLex-999 data")
    
    # WordSim-353
    try:
        # Note: The actual WordSim-353 URL might be different
        sample_wordsim = """Word 1\tWord 2\tHuman (mean)
love\tsex\t6.77
tiger\tcat\t7.35
book\tpaper\t7.46
computer\tkeyboard\t7.62
money\tcash\t9.15"""
        with open('wordsim353.txt', 'w') as f:
            f.write(sample_wordsim)
        print("✅ Created sample WordSim-353 data")
    except Exception as e:
        print(f"⚠️  Could not create WordSim-353: {e}")

def check_data_directories():
    """Check if data directories exist"""
    print("📁 Checking data directories...")
    
    required_dirs = [
        "Data/eng_news_2020_300K",
        "Data/hin_news_2020_300K"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("⚠️  Missing data directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\n📝 Please download the corpora from:")
        print("   English: https://wortschatz.uni-leipzig.de/en/download/English")
        print("   Hindi: https://wortschatz.uni-leipzig.de/en/download/Hindi")
        return False
    else:
        print("✅ All data directories found!")
        return True

def create_output_directories():
    """Create directories for output files"""
    print("📂 Creating output directories...")
    
    output_dirs = [
        "results",
        "visualizations",
        "models"
    ]
    
    for dir_name in output_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    print("✅ Output directories created!")

def main():
    """Main setup function"""
    print("🚀 Setting up Language Representations project...")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation.")
        return
    
    # Check data directories
    data_available = check_data_directories()
    
    # Download evaluation datasets
    download_evaluation_datasets()
    
    # Create output directories
    create_output_directories()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    
    if data_available:
        print("\n✅ You're ready to run the notebooks!")
        print("\n📚 Recommended order:")
        print("   1. Part1_Dense_Representations.ipynb")
        print("   2. Part1_Neural_Embeddings_Comparison.ipynb")
        print("   3. Part2_Cross_Lingual_Alignment.ipynb")
        print("   4. Bonus_Harmful_Associations.ipynb")
    else:
        print("\n⚠️  Please download the required corpora before running the notebooks.")
    
    print("\n🔧 To start Jupyter:")
    print("   jupyter notebook")

if __name__ == "__main__":
    main()
