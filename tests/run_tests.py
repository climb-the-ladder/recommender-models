import unittest
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Discover and run all tests
if __name__ == "__main__":
    # Discover all tests in the current directory
    test_suite = unittest.defaultTestLoader.discover(
        os.path.dirname(os.path.abspath(__file__)),
        pattern="test_*.py"
    )
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return non-zero exit code if tests failed, for CI/CD pipelines
    sys.exit(0 if result.wasSuccessful() else 1) 