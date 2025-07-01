#!/usr/bin/env python3
"""
ONNX to MAX Conversion Script

This script converts all ONNX models to MAX format for optimized inference.
"""

import os
import subprocess
import json
from pathlib import Path
import time

class MAXConverter:
    """Handles conversion of ONNX models to MAX format."""
    
    def __init__(self):
        self.workspace_root = Path(__file__).parent.parent
        self.onnx_dir = self.workspace_root / "models" / "onnx"
        self.max_dir = self.workspace_root / "models" / "max"
        
        # Ensure MAX directory exists
        self.max_dir.mkdir(exist_ok=True)
        
        self.conversion_results = {}
    
    def convert_model(self, onnx_file: Path, output_name: str = None) -> bool:
        """
        Convert a single ONNX model to MAX format.
        
        Args:
            onnx_file: Path to ONNX model file
            output_name: Optional custom output name
            
        Returns:
            True if conversion successful, False otherwise
        """
        if not onnx_file.exists():
            print(f"✗ ONNX file not found: {onnx_file}")
            return False
        
        # Determine output file name
        if output_name is None:
            output_name = onnx_file.stem
        
        max_file = self.max_dir / f"{output_name}.maxgraph"
        
        print(f"Converting {onnx_file.name} to MAX format...")
        
        try:
            # Run MAX convert command
            cmd = [
                "max", "convert",
                str(onnx_file),
                "--output-file", str(max_file)
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            conversion_time = time.time() - start_time
            
            if result.returncode == 0:
                file_size = max_file.stat().st_size / 1024  # KB
                print(f"✓ {output_name}.maxgraph created ({file_size:.1f} KB, {conversion_time:.1f}s)")
                
                self.conversion_results[output_name] = {
                    "status": "success",
                    "input_file": str(onnx_file),
                    "output_file": str(max_file),
                    "file_size_kb": file_size,
                    "conversion_time_s": conversion_time
                }
                return True
            else:
                print(f"✗ Conversion failed for {onnx_file.name}")
                print(f"Error: {result.stderr}")
                
                self.conversion_results[output_name] = {
                    "status": "failed",
                    "error": result.stderr,
                    "input_file": str(onnx_file)
                }
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Conversion timeout for {onnx_file.name}")
            self.conversion_results[output_name] = {
                "status": "timeout",
                "input_file": str(onnx_file)
            }
            return False
        except Exception as e:
            print(f"✗ Conversion error for {onnx_file.name}: {e}")
            self.conversion_results[output_name] = {
                "status": "error",
                "error": str(e),
                "input_file": str(onnx_file)
            }
            return False
    
    def convert_all_models(self):
        """Convert all ONNX models to MAX format."""
        print("Starting ONNX to MAX conversion...")
        print("="*60)
        
        # Get all ONNX files
        onnx_files = list(self.onnx_dir.glob("*.onnx"))
        
        if not onnx_files:
            print("No ONNX files found for conversion")
            return
        
        print(f"Found {len(onnx_files)} ONNX models to convert:")
        for onnx_file in onnx_files:
            file_size = onnx_file.stat().st_size / 1024
            print(f"  • {onnx_file.name} ({file_size:.1f} KB)")
        
        print()
        
        # Convert each model
        successful_conversions = 0
        for onnx_file in onnx_files:
            if self.convert_model(onnx_file):
                successful_conversions += 1
            print()
        
        # Summary
        print("="*60)
        print("MAX Conversion Summary:")
        print(f"Total models: {len(onnx_files)}")
        print(f"Successful: {successful_conversions}")
        print(f"Failed: {len(onnx_files) - successful_conversions}")
        
        # List converted files
        max_files = list(self.max_dir.glob("*.maxgraph"))
        if max_files:
            print(f"\nConverted MAX models:")
            for max_file in max_files:
                file_size = max_file.stat().st_size / 1024
                print(f"  • {max_file.name} ({file_size:.1f} KB)")
        
        # Save conversion metadata
        self.save_conversion_metadata()
    
    def save_conversion_metadata(self):
        """Save conversion results to metadata file."""
        metadata = {
            "conversion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "max_version": self.get_max_version(),
            "conversion_results": self.conversion_results
        }
        
        metadata_file = self.workspace_root / "models" / "max_conversion_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Conversion metadata saved to: {metadata_file}")
    
    def get_max_version(self) -> str:
        """Get MAX version."""
        try:
            result = subprocess.run(["max", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "unknown"
    
    def verify_conversions(self):
        """Verify that all conversions were successful."""
        print("\nVerifying MAX model conversions...")
        
        max_files = list(self.max_dir.glob("*.maxgraph"))
        
        for max_file in max_files:
            try:
                # Basic file validation
                if max_file.stat().st_size > 0:
                    print(f"✓ {max_file.name} - Valid file")
                else:
                    print(f"✗ {max_file.name} - Empty file")
            except Exception as e:
                print(f"✗ {max_file.name} - Error: {e}")

def main():
    """Main conversion function."""
    converter = MAXConverter()
    
    try:
        converter.convert_all_models()
        converter.verify_conversions()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Create Mojo wrappers for MAX models")
        print("2. Implement performance benchmarking")
        print("3. Create validation tests")
        print("4. Deploy optimized inference server")
        
        return 0
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
