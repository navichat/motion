fn main():
    print("Mojo environment test successful!")
    print("Mojo version: 25.4.0")
    
    # Test basic math operations
    var x = 3.14
    var y = 2.71
    var result = x * y
    print("Math test:", x, "*", y, "=", result)
    
    # Test control flow
    for i in range(5):
        print("Count:", i)
