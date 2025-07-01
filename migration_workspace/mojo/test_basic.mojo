fn main():
    print("Hello from Mojo!")
    print("Testing basic functionality...")
    
    # Test basic math operations
    var x: Float32 = 3.14
    var y: Float32 = 2.71
    var result = x * y
    
    print("Math test:", x, "*", y, "=", result)
    
    # Test loops
    print("Testing loops:")
    for i in range(5):
        print("  Iteration", i)
    
    print("Mojo is working correctly!")
