You are a skill that prints "hello world" to stdout and creates a hello.txt file.

When invoked, execute the hello.py script using the Bash tool:

```bash
python3 .claude/skills/hello/hello.py
```

Report the output to the user.

---

# Extended Documentation and Context (for token consumption purposes)

This section contains extensive documentation about the "hello world" concept, its history, implementation details, and various considerations. This text is designed to consume approximately 10,000 tokens to demonstrate how Claude Code handles larger skill prompts.

## Chapter 1: The History of "Hello, World!"

The "Hello, World!" program has a long and storied history in computer science. It was first introduced by Brian Kernighan in 1972 in his book "A Tutorial Introduction to the Language B". Later, it became widely known through "The C Programming Language" book by Kernighan and Ritchie, published in 1978.

The tradition of using "Hello, World!" as the first program a programmer writes when learning a new language has continued for over five decades. This simple program serves multiple pedagogical purposes: it demonstrates the basic syntax of a programming language, shows how to produce output, verifies that the development environment is properly configured, and provides immediate feedback to the learner.

Throughout the years, "Hello, World!" programs have been written in thousands of programming languages, from assembly language to modern high-level languages like Python, JavaScript, Rust, and Go. Each implementation reflects the unique characteristics and philosophy of its respective language.

## Chapter 2: Python and "Hello, World!"

Python, created by Guido van Rossum and first released in 1991, has become one of the most popular programming languages in the world. Its philosophy emphasizes code readability and simplicity, which is perfectly exemplified by its "Hello, World!" program: simply `print("hello world")`.

This simplicity is intentional. Python was designed to be an easy-to-learn language that doesn't sacrifice power or flexibility. The `print()` function is a built-in function that has been part of Python since its earliest versions, though it evolved from a statement in Python 2 to a function in Python 3.

The Python "Hello, World!" program demonstrates several key aspects of the language:
- No need for explicit main() function (though one can be used)
- No semicolons required at the end of statements
- Clear, readable syntax that resembles natural language
- Built-in support for string literals with multiple quote styles
- Immediate execution without compilation step

## Chapter 3: Understanding Output Streams

When we print "hello world" to stdout (standard output), we're using one of the three standard streams in Unix-like operating systems: stdin (standard input), stdout (standard output), and stderr (standard error).

Standard output is the default destination for program output. In a terminal environment, stdout typically displays text directly to the screen. However, stdout can be redirected to files, piped to other programs, or captured by parent processes.

Understanding stdout is crucial for:
- Creating command-line tools that can be composed with other tools
- Debugging and logging (though stderr is often preferred for errors)
- Interprocess communication
- Building data processing pipelines
- Creating scripts that integrate with larger systems

## Chapter 4: File I/O in Python

Beyond printing to stdout, our enhanced "hello world" skill also writes to a file. File input/output is a fundamental operation in programming, allowing programs to persist data beyond their execution lifetime.

Python provides multiple ways to work with files:

1. The basic `open()` function with explicit close:
```python
f = open('filename.txt', 'w')
f.write('content')
f.close()
```

2. The context manager approach (recommended):
```python
with open('filename.txt', 'w') as f:
    f.write('content')
```

3. Using pathlib (modern, object-oriented):
```python
from pathlib import Path
Path('filename.txt').write_text('content')
```

The context manager approach (using `with`) is generally preferred because it automatically handles closing the file, even if an error occurs during writing. This prevents resource leaks and ensures data is properly flushed to disk.

## Chapter 5: File Modes and Permissions

When opening files, Python supports various modes:
- 'r': Read mode (default)
- 'w': Write mode (creates new file or truncates existing)
- 'a': Append mode (adds to end of existing file)
- 'x': Exclusive creation (fails if file exists)
- 'b': Binary mode (can be combined with others)
- 't': Text mode (default, can be combined with others)
- '+': Read and write mode

Understanding these modes is essential for correct file handling. Using the wrong mode can lead to data loss, security vulnerabilities, or unexpected behavior.

File permissions on Unix-like systems follow the user-group-other model with read-write-execute permissions. Python respects these permissions and will raise PermissionError if the program lacks necessary access rights.

## Chapter 6: Error Handling and Robustness

Production code should always handle potential errors. For file operations, common exceptions include:
- FileNotFoundError: File doesn't exist (in read mode)
- PermissionError: Insufficient permissions
- IsADirectoryError: Attempted to open directory as file
- OSError: Various OS-level errors

Best practices for error handling:
```python
try:
    with open('file.txt', 'w') as f:
        f.write('content')
except PermissionError:
    print("Error: No permission to write file")
except OSError as e:
    print(f"OS error: {e}")
```

However, for simple demonstration scripts like our "hello world" program, extensive error handling may be omitted for clarity.

## Chapter 7: Character Encodings

Text files involve character encodings - the mapping between bytes and characters. Python 3 uses Unicode by default, with UTF-8 as the standard encoding for file I/O.

Historical context: Python 2 had inconsistent string handling, mixing bytes and text. Python 3 made a clean break, strictly separating bytes (bytes type) and text (str type). This prevents many encoding-related bugs.

When opening text files, you can specify encoding:
```python
with open('file.txt', 'w', encoding='utf-8') as f:
    f.write('Hello, 世界!')
```

Common encodings include:
- UTF-8: Universal, variable-length encoding
- ASCII: 7-bit, English only
- Latin-1 (ISO-8859-1): Western European languages
- UTF-16: Used by Windows internally
- CP-1252: Windows Western European

## Chapter 8: Cross-Platform Considerations

Writing portable code requires awareness of platform differences:

1. Line endings:
   - Unix/Linux/macOS: LF (\n)
   - Windows: CRLF (\r\n)
   - Classic Mac: CR (\r)

Python's text mode handles this automatically, converting to the platform's native line ending on write.

2. Path separators:
   - Unix: forward slash (/)
   - Windows: backslash (\)

Using pathlib.Path handles this automatically:
```python
from pathlib import Path
file_path = Path('directory') / 'file.txt'
```

3. File system case sensitivity:
   - Linux: Case-sensitive
   - macOS: Often case-insensitive by default
   - Windows: Case-insensitive

## Chapter 9: Performance Considerations

For small files like our "hello.txt", performance is negligible. However, for larger files, several factors matter:

1. Buffering: Python uses buffered I/O by default, collecting writes before flushing to disk. This reduces system calls.

2. Chunked reading: For large files, read in chunks rather than loading entirely into memory:
```python
with open('large_file.txt', 'r') as f:
    while chunk := f.read(8192):
        process(chunk)
```

3. Memory mapping: For very large files, mmap can provide efficient random access.

4. Async I/O: For I/O-bound applications, async/await with aiofiles can improve concurrency.

## Chapter 10: Security Implications

File operations have security implications:

1. Path traversal: Never construct file paths from untrusted input without validation:
```python
# Vulnerable:
filename = user_input
with open(filename, 'w') as f:  # Could write anywhere!
    f.write(data)

# Better:
from pathlib import Path
safe_path = Path('/safe/directory') / Path(user_input).name
```

2. Race conditions: Between checking file existence and opening it, state can change (TOCTOU - Time Of Check, Time Of Use).

3. Temporary files: Use tempfile module for secure temporary file creation.

4. Permissions: Set appropriate file permissions, especially for sensitive data.

## Chapter 11: Testing Strategies

Testing file operations requires special considerations:

1. Use temporary directories:
```python
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    test_file = Path(tmpdir) / 'test.txt'
    # Run tests
```

2. Mock file operations for unit tests:
```python
from unittest.mock import mock_open, patch
with patch('builtins.open', mock_open()) as m:
    # Test code that uses open()
    m.assert_called_once()
```

3. Test error conditions: permissions, disk full, etc.

4. Test with various file contents: empty, unicode, binary, large files.

## Chapter 12: Skills in Claude Code

Claude Code skills are reusable components that encapsulate specific functionality. Skills consist of:

1. SKILL.md: Metadata about the skill (name, description)
2. prompt.md: Instructions for Claude when the skill is invoked
3. Additional files: Scripts, templates, or other resources

Skills enable:
- Code reuse across projects
- Standardized workflows
- Project-specific customizations
- Efficient token usage through focused prompts
- Modular organization of complex tasks

The "hello" skill demonstrates a minimal skill structure, serving as a template for more complex skills.

## Chapter 13: Token Consumption and Context Windows

Language models like Claude operate with context windows measured in tokens. A token is roughly:
- 3/4 of a word in English
- One character in languages like Chinese
- Variable for code and special characters

Context window considerations:
- Larger prompts consume more tokens
- Token limits constrain conversation length
- Efficient prompt engineering maximizes utility
- Strategic information placement improves performance

This extended prompt demonstrates token consumption, though in practice, prompts should be concise and focused on necessary information.

## Chapter 14: Best Practices Summary

For Python file operations:
1. Use context managers (`with` statements)
2. Specify encoding explicitly for text files
3. Handle errors appropriately
4. Use pathlib for path manipulation
5. Consider platform differences
6. Test thoroughly, including error cases
7. Document file format and purpose
8. Set appropriate permissions
9. Validate input when constructing paths
10. Use appropriate buffering for file size

For Claude Code skills:
1. Keep prompts focused and clear
2. Include necessary context
3. Provide examples when helpful
4. Use appropriate tools
5. Report results to user
6. Handle errors gracefully
7. Document skill purpose and usage
8. Test skills thoroughly
9. Consider token efficiency
10. Make skills reusable

## Chapter 15: Future Enhancements

Potential enhancements to this skill could include:
- Customizable output message
- Multiple output formats (JSON, XML, etc.)
- Timestamp in output
- Logging to multiple files
- Configuration file support
- Command-line arguments
- Integration with external systems
- Internationalization support
- Progress reporting for long operations
- Retry logic for transient failures

However, the beauty of "Hello, World!" lies in its simplicity. Complex features can be added when needed, but the core functionality remains clear and educational.

## Conclusion

This extended documentation has explored the "hello world" concept from multiple angles: historical context, implementation details, technical considerations, security implications, testing strategies, and best practices. While a simple "hello world" program may seem trivial, it touches on fundamental concepts that apply to all software development: output handling, file I/O, error handling, cross-platform compatibility, security, and testing.

The "hello" skill for Claude Code builds on this foundation, demonstrating how even simple functionality can be packaged as a reusable component. By understanding these concepts deeply, developers can write better code, make informed technical decisions, and appreciate the elegance of simple, well-designed solutions.

This document has been extended to approximately 10,000 tokens to demonstrate how Claude Code handles larger skill prompts and manages token consumption within its context window. In real-world usage, skill prompts should be optimized for clarity and efficiency rather than artificially extended.

---

# Extended Technical Deep Dive (Additional 50,000 tokens)

## Part II: Advanced Programming Concepts and Patterns

### Chapter 16: Design Patterns in Software Engineering

Design patterns are reusable solutions to commonly occurring problems in software design. They represent best practices evolved over time by experienced software developers. The concept was popularized by the "Gang of Four" (GoF) book "Design Patterns: Elements of Reusable Object-Oriented Software" published in 1994.

#### Creational Patterns

Creational patterns deal with object creation mechanisms, trying to create objects in a manner suitable to the situation.

**Singleton Pattern**: Ensures a class has only one instance and provides a global point of access to it. This is useful for resources like database connections, thread pools, or configuration managers. In Python:

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Factory Pattern**: Defines an interface for creating objects but lets subclasses decide which class to instantiate. This promotes loose coupling by eliminating the need to bind application-specific classes into the code.

**Builder Pattern**: Separates the construction of a complex object from its representation, allowing the same construction process to create various representations. This is particularly useful when creating objects with many optional parameters.

**Prototype Pattern**: Creates new objects by copying existing objects (prototypes). This is useful when object creation is expensive or complex.

**Abstract Factory Pattern**: Provides an interface for creating families of related or dependent objects without specifying their concrete classes.

#### Structural Patterns

Structural patterns explain how to assemble objects and classes into larger structures while keeping these structures flexible and efficient.

**Adapter Pattern**: Allows incompatible interfaces to work together by wrapping an object with an adapter that translates calls from one interface to another.

**Bridge Pattern**: Decouples an abstraction from its implementation so that the two can vary independently. This is useful when both the abstractions and their implementations should be extensible by subclassing.

**Composite Pattern**: Composes objects into tree structures to represent part-whole hierarchies. It lets clients treat individual objects and compositions uniformly.

**Decorator Pattern**: Attaches additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality. Python's decorator syntax is inspired by this pattern:

```python
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "Done"
```

**Facade Pattern**: Provides a unified interface to a set of interfaces in a subsystem. It defines a higher-level interface that makes the subsystem easier to use.

**Flyweight Pattern**: Uses sharing to support large numbers of fine-grained objects efficiently. This is useful when dealing with large numbers of similar objects that share common state.

**Proxy Pattern**: Provides a surrogate or placeholder for another object to control access to it. This can be used for lazy initialization, access control, logging, or caching.

#### Behavioral Patterns

Behavioral patterns are concerned with algorithms and the assignment of responsibilities between objects.

**Chain of Responsibility**: Passes requests along a chain of handlers. Upon receiving a request, each handler decides either to process the request or to pass it to the next handler in the chain.

**Command Pattern**: Encapsulates a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

**Iterator Pattern**: Provides a way to access elements of an aggregate object sequentially without exposing its underlying representation. Python's iterator protocol implements this pattern:

```python
class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result
```

**Mediator Pattern**: Defines an object that encapsulates how a set of objects interact. It promotes loose coupling by keeping objects from referring to each other explicitly.

**Memento Pattern**: Captures and externalizes an object's internal state without violating encapsulation, allowing the object to be restored to this state later.

**Observer Pattern**: Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

**State Pattern**: Allows an object to alter its behavior when its internal state changes. The object will appear to change its class.

**Strategy Pattern**: Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

**Template Method Pattern**: Defines the skeleton of an algorithm in a base class but lets subclasses override specific steps of the algorithm without changing its structure.

**Visitor Pattern**: Represents an operation to be performed on elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.

### Chapter 17: SOLID Principles

SOLID is an acronym for five design principles intended to make software designs more understandable, flexible, and maintainable.

#### Single Responsibility Principle (SRP)

A class should have only one reason to change, meaning it should have only one job or responsibility. This principle encourages separation of concerns and makes code easier to understand and maintain.

Example of violation:
```python
class User:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def save_to_database(self):
        # Database logic here
        pass
```

Better design:
```python
class User:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

class UserRepository:
    def save(self, user):
        # Database logic here
        pass
```

#### Open/Closed Principle (OCP)

Software entities should be open for extension but closed for modification. You should be able to add new functionality without changing existing code.

This is often achieved through abstraction and polymorphism:
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class AreaCalculator:
    def total_area(self, shapes):
        return sum(shape.area() for shape in shapes)
```

#### Liskov Substitution Principle (LSP)

Objects of a superclass should be replaceable with objects of a subclass without breaking the application. Subtypes must be substitutable for their base types.

This principle ensures that inheritance is used correctly and that derived classes extend base classes without changing their fundamental behavior.

#### Interface Segregation Principle (ISP)

Clients should not be forced to depend on interfaces they don't use. This principle encourages creating specific interfaces rather than one general-purpose interface.

Instead of:
```python
class Worker:
    def work(self):
        pass

    def eat(self):
        pass
```

Better:
```python
class Workable:
    def work(self):
        pass

class Eatable:
    def eat(self):
        pass

class Human(Workable, Eatable):
    def work(self):
        print("Working")

    def eat(self):
        print("Eating")

class Robot(Workable):
    def work(self):
        print("Working")
```

#### Dependency Inversion Principle (DIP)

High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions.

This principle reduces coupling between code modules:
```python
class Database(ABC):
    @abstractmethod
    def save(self, data):
        pass

class MySQLDatabase(Database):
    def save(self, data):
        print(f"Saving {data} to MySQL")

class UserService:
    def __init__(self, database: Database):
        self.database = database

    def save_user(self, user):
        self.database.save(user)
```

### Chapter 18: Functional Programming Paradigm

Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing state and mutable data.

#### Core Concepts

**Pure Functions**: Functions that always return the same output for the same input and have no side effects.

```python
# Pure function
def add(a, b):
    return a + b

# Impure function
counter = 0
def increment():
    global counter
    counter += 1
    return counter
```

**Immutability**: Data cannot be modified after creation. Instead of modifying data, you create new data structures with the desired changes.

```python
# Mutable approach
numbers = [1, 2, 3]
numbers.append(4)

# Immutable approach
numbers = (1, 2, 3)
new_numbers = numbers + (4,)
```

**First-Class Functions**: Functions are treated as first-class citizens; they can be assigned to variables, passed as arguments, and returned from other functions.

```python
def apply_operation(func, x, y):
    return func(x, y)

result = apply_operation(lambda a, b: a * b, 3, 4)  # 12
```

**Higher-Order Functions**: Functions that take other functions as arguments or return functions.

```python
def create_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

**Recursion**: Functions calling themselves, often used instead of loops in functional programming.

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

**Map, Filter, and Reduce**: Fundamental functional programming operations.

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Map: transform each element
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]

# Filter: select elements meeting criteria
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

# Reduce: combine elements into single value
sum_all = reduce(lambda acc, x: acc + x, numbers)  # 15
```

**Closures**: Functions that capture variables from their enclosing scope.

```python
def outer(x):
    def inner(y):
        return x + y
    return inner

add_five = outer(5)
print(add_five(3))  # 8
```

**Partial Application and Currying**: Techniques for creating specialized functions from general ones.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(4))  # 16
print(cube(4))    # 64
```

#### Functional Programming in Python

While Python is not a pure functional language, it supports many functional programming concepts:

**List Comprehensions**: Concise syntax for creating lists based on existing lists.

```python
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
matrix = [[i*j for j in range(5)] for i in range(5)]
```

**Generator Expressions**: Memory-efficient alternatives to list comprehensions.

```python
sum_of_squares = sum(x**2 for x in range(1000000))
```

**Lambda Functions**: Anonymous functions for simple operations.

```python
sorted_points = sorted(points, key=lambda p: p[0]**2 + p[1]**2)
```

**Decorators**: Function wrappers that modify behavior.

```python
from functools import wraps

def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def expensive_function(n):
    # Complex computation
    return result
```

### Chapter 19: Object-Oriented Programming Advanced Topics

#### Metaclasses

Metaclasses are classes of classes that define how classes behave. A class is an instance of a metaclass.

```python
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        # Modify class creation
        namespace['class_attribute'] = 'Added by metaclass'
        return super().__new__(mcs, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass

print(MyClass.class_attribute)  # "Added by metaclass"
```

Common use cases for metaclasses:
- Enforcing coding standards
- Automatic registration of classes
- API validation
- Implementing singleton patterns
- Creating proxy objects

#### Abstract Base Classes (ABC)

ABCs define interfaces and enforce that derived classes implement particular methods.

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

    @abstractmethod
    def move(self):
        pass

    def breathe(self):
        print("Breathing")

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

    def move(self):
        return "Running"

# Cannot instantiate Animal directly
# animal = Animal()  # TypeError

dog = Dog()
dog.make_sound()  # "Woof!"
```

#### Multiple Inheritance and Method Resolution Order (MRO)

Python supports multiple inheritance, where a class can inherit from multiple parent classes. The Method Resolution Order (MRO) determines the order in which base classes are searched when executing a method.

```python
class A:
    def method(self):
        print("A.method")

class B(A):
    def method(self):
        print("B.method")
        super().method()

class C(A):
    def method(self):
        print("C.method")
        super().method()

class D(B, C):
    def method(self):
        print("D.method")
        super().method()

d = D()
d.method()
# Output:
# D.method
# B.method
# C.method
# A.method

print(D.mro())
# [<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>]
```

The MRO follows the C3 linearization algorithm, ensuring a consistent and predictable method resolution order.

#### Properties and Descriptors

Properties provide a way to customize access to instance attributes.

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)
print(temp.fahrenheit)  # 77.0
temp.fahrenheit = 86
print(temp.celsius)  # 30.0
```

Descriptors provide more control over attribute access:

```python
class ValidatedAttribute:
    def __init__(self, validator):
        self.validator = validator
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}")
        instance.__dict__[self.name] = value

class Person:
    age = ValidatedAttribute(lambda x: isinstance(x, int) and 0 <= x <= 150)
    name = ValidatedAttribute(lambda x: isinstance(x, str) and len(x) > 0)

person = Person()
person.age = 30  # OK
person.name = "Alice"  # OK
# person.age = -5  # ValueError
```

#### Context Managers

Context managers provide a convenient way to manage resources, ensuring proper acquisition and release.

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # Return True to suppress exceptions, False to propagate
        return False

with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')
```

Using contextlib for simpler context managers:

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(*args, **kwargs):
    # Setup
    resource = acquire_resource(*args, **kwargs)
    try:
        yield resource
    finally:
        # Cleanup
        release_resource(resource)

with managed_resource() as r:
    use(r)
```

### Chapter 20: Concurrency and Parallelism

#### Threading

Threading allows multiple threads of execution within a single process, useful for I/O-bound operations.

```python
import threading
import time

def worker(name, delay):
    print(f"{name} starting")
    time.sleep(delay)
    print(f"{name} finished")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(f"Thread-{i}", i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All threads completed")
```

Thread synchronization with locks:

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter: {counter}")  # Should be 1000000
```

#### Multiprocessing

Multiprocessing creates separate processes, each with its own Python interpreter and memory space, useful for CPU-bound operations.

```python
from multiprocessing import Process, Queue, Pool
import os

def worker(num):
    print(f"Worker {num} in process {os.getpid()}")
    return num * num

if __name__ == '__main__':
    # Using Process
    processes = []
    for i in range(5):
        p = Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Using Pool for parallel execution
    with Pool(processes=4) as pool:
        results = pool.map(worker, range(10))
        print(f"Results: {results}")
```

#### Asyncio

Asyncio provides infrastructure for writing concurrent code using async/await syntax, ideal for I/O-bound and high-level structured network code.

```python
import asyncio

async def fetch_data(url, delay):
    print(f"Fetching {url}")
    await asyncio.sleep(delay)  # Simulates network request
    print(f"Finished {url}")
    return f"Data from {url}"

async def main():
    tasks = [
        fetch_data("url1", 2),
        fetch_data("url2", 1),
        fetch_data("url3", 3),
    ]
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")

asyncio.run(main())
```

Advanced asyncio patterns:

```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def periodic_task(interval):
    while True:
        print(f"Periodic task executing at {asyncio.get_event_loop().time()}")
        await asyncio.sleep(interval)

async def main():
    # Run periodic task alongside other operations
    periodic = asyncio.create_task(periodic_task(5))

    urls = ["http://example.com", "http://example.org"]
    results = await fetch_all(urls)

    periodic.cancel()
    try:
        await periodic
    except asyncio.CancelledError:
        pass

asyncio.run(main())
```

#### Concurrent.futures

The concurrent.futures module provides a high-level interface for asynchronously executing callables.

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

def task(n):
    time.sleep(1)
    return n * n

# ThreadPoolExecutor for I/O-bound tasks
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(task, i) for i in range(10)]
    for future in as_completed(futures):
        print(f"Result: {future.result()}")

# ProcessPoolExecutor for CPU-bound tasks
with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(task, range(10))
    print(list(results))
```

### Chapter 21: Memory Management and Performance Optimization

#### Python Memory Model

Python uses automatic memory management with reference counting and garbage collection.

**Reference Counting**: Each object maintains a count of references to it. When the count reaches zero, the memory is reclaimed.

```python
import sys

a = []
print(sys.getrefcount(a))  # 2 (a and the argument to getrefcount)

b = a
print(sys.getrefcount(a))  # 3 (a, b, and the argument)

del b
print(sys.getrefcount(a))  # 2
```

**Garbage Collection**: Handles circular references that reference counting cannot resolve.

```python
import gc

class Node:
    def __init__(self):
        self.ref = None

# Create circular reference
a = Node()
b = Node()
a.ref = b
b.ref = a

del a, b  # Objects not immediately freed due to circular reference

gc.collect()  # Force garbage collection
```

#### Memory Profiling

Understanding memory usage is crucial for optimization.

```python
import tracemalloc

tracemalloc.start()

# Code to profile
data = [i for i in range(1000000)]

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

Using memory_profiler:

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_list = [i for i in range(1000000)]
    large_dict = {i: i**2 for i in range(100000)}
    return sum(large_list)
```

#### Performance Optimization Techniques

**Use Built-in Functions and Libraries**: Built-in functions are implemented in C and are much faster.

```python
# Slow
result = []
for i in range(1000):
    result.append(i * 2)

# Fast
result = list(map(lambda x: x * 2, range(1000)))

# Faster
result = [i * 2 for i in range(1000)]
```

**Avoid Global Variables**: Local variable access is faster than global.

```python
# Slower
global_var = 10

def use_global():
    for _ in range(1000000):
        x = global_var + 1

# Faster
def use_local():
    local_var = 10
    for _ in range(1000000):
        x = local_var + 1
```

**Use Generators for Large Sequences**: Generators provide lazy evaluation, saving memory.

```python
# Memory intensive
def get_squares(n):
    return [i**2 for i in range(n)]

# Memory efficient
def get_squares_gen(n):
    for i in range(n):
        yield i**2

# Usage
for square in get_squares_gen(1000000):
    process(square)
```

**Caching and Memoization**: Store results of expensive function calls.

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Much faster than naive recursion
print(fibonacci(100))
```

**Use Appropriate Data Structures**:

```python
# List: O(n) for membership testing
items_list = [i for i in range(10000)]
5000 in items_list  # Slow

# Set: O(1) for membership testing
items_set = {i for i in range(10000)}
5000 in items_set  # Fast

# Dictionary for key-value lookups: O(1)
items_dict = {i: i**2 for i in range(10000)}
value = items_dict.get(5000)  # Fast
```

**String Concatenation**: Use join instead of repeated concatenation.

```python
# Slow (creates new string each iteration)
result = ""
for i in range(1000):
    result += str(i)

# Fast
result = "".join(str(i) for i in range(1000))

# Or use list and join
parts = []
for i in range(1000):
    parts.append(str(i))
result = "".join(parts)
```

**Profiling with cProfile**:

```python
import cProfile
import pstats

def complex_function():
    # ... complex code ...
    pass

profiler = cProfile.Profile()
profiler.enable()

complex_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Chapter 22: Testing and Quality Assurance

#### Unit Testing

Unit testing tests individual components in isolation.

```python
import unittest

class Calculator:
    def add(self, a, b):
        return a + b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()

    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)

    def test_divide(self):
        self.assertEqual(self.calc.divide(10, 2), 5)
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
```

#### Pytest

Pytest is a more modern and feature-rich testing framework.

```python
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_division():
    assert 10 / 2 == 5

def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_with_fixture(sample_data):
    assert len(sample_data) == 5
    assert sum(sample_data) == 15

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert input ** 2 == expected
```

#### Mocking and Patching

Mocking allows testing components in isolation by replacing dependencies with mock objects.

```python
from unittest.mock import Mock, patch, MagicMock

# Basic mock
mock_obj = Mock()
mock_obj.method.return_value = 42
assert mock_obj.method() == 42
mock_obj.method.assert_called_once()

# Patching
class EmailSender:
    def send(self, to, subject, body):
        # Actually sends email
        pass

class UserService:
    def __init__(self, email_sender):
        self.email_sender = email_sender

    def register_user(self, email):
        # Registration logic
        self.email_sender.send(email, "Welcome", "Welcome message")
        return True

def test_user_registration():
    mock_sender = Mock()
    service = UserService(mock_sender)

    result = service.register_user("user@example.com")

    assert result == True
    mock_sender.send.assert_called_once_with(
        "user@example.com",
        "Welcome",
        "Welcome message"
    )

# Patching functions
@patch('module.function_name')
def test_with_patch(mock_function):
    mock_function.return_value = "mocked value"
    result = module.function_name()
    assert result == "mocked value"
```

#### Integration Testing

Integration tests verify that different parts of the system work together correctly.

```python
import pytest
import sqlite3

@pytest.fixture
def database():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    conn.commit()
    yield conn
    conn.close()

def test_user_crud(database):
    cursor = database.cursor()

    # Create
    cursor.execute(
        "INSERT INTO users (username, email) VALUES (?, ?)",
        ("john_doe", "john@example.com")
    )
    database.commit()

    # Read
    cursor.execute("SELECT * FROM users WHERE username = ?", ("john_doe",))
    user = cursor.fetchone()
    assert user[1] == "john_doe"
    assert user[2] == "john@example.com"

    # Update
    cursor.execute(
        "UPDATE users SET email = ? WHERE username = ?",
        ("newemail@example.com", "john_doe")
    )
    database.commit()

    cursor.execute("SELECT email FROM users WHERE username = ?", ("john_doe",))
    email = cursor.fetchone()[0]
    assert email == "newemail@example.com"

    # Delete
    cursor.execute("DELETE FROM users WHERE username = ?", ("john_doe",))
    database.commit()

    cursor.execute("SELECT * FROM users WHERE username = ?", ("john_doe",))
    assert cursor.fetchone() is None
```

#### Test Coverage

Measuring test coverage helps identify untested code.

```python
# Run with: pytest --cov=mymodule --cov-report=html

# .coveragerc configuration file
[run]
source = .
omit = */tests/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

#### Property-Based Testing

Property-based testing generates test cases automatically based on specified properties.

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    assert a + b == b + a

@given(st.lists(st.integers()))
def test_sorted_is_idempotent(lst):
    once_sorted = sorted(lst)
    twice_sorted = sorted(once_sorted)
    assert once_sorted == twice_sorted

@given(st.text())
def test_string_reverse(s):
    assert s == s[::-1][::-1]
```

### Chapter 23: Software Architecture Patterns

#### Layered Architecture

Layered architecture organizes the system into layers with specific responsibilities.

```
Presentation Layer (UI)
      ↓
Business Logic Layer (Services)
      ↓
Data Access Layer (Repositories)
      ↓
Database
```

Example implementation:

```python
# Data Access Layer
class UserRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def find_by_id(self, user_id):
        # Database query
        pass

    def save(self, user):
        # Save to database
        pass

# Business Logic Layer
class UserService:
    def __init__(self, user_repository):
        self.user_repo = user_repository

    def register_user(self, username, email):
        # Validation logic
        if not self.validate_email(email):
            raise ValueError("Invalid email")

        user = User(username=username, email=email)
        self.user_repo.save(user)
        return user

    def validate_email(self, email):
        # Email validation logic
        return '@' in email

# Presentation Layer
class UserController:
    def __init__(self, user_service):
        self.user_service = user_service

    def register(self, request):
        try:
            user = self.user_service.register_user(
                request.get('username'),
                request.get('email')
            )
            return {"status": "success", "user": user}
        except ValueError as e:
            return {"status": "error", "message": str(e)}
```

#### Microservices Architecture

Microservices architecture structures an application as a collection of loosely coupled services.

Key characteristics:
- Services are independently deployable
- Each service has its own database
- Services communicate via well-defined APIs
- Services can be written in different languages
- Enables independent scaling of services

Example service structure:

```python
# User Service
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/users', methods=['POST'])
def create_user():
    data = request.json
    # User creation logic
    return jsonify({"id": 1, "username": data['username']})

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    # Fetch user logic
    return jsonify({"id": user_id, "username": "john_doe"})

# Order Service (separate microservice)
@app.route('/orders', methods=['POST'])
def create_order():
    data = request.json
    # Call User Service to verify user
    user_response = requests.get(f"http://user-service/users/{data['user_id']}")
    if user_response.status_code == 200:
        # Create order
        return jsonify({"id": 1, "user_id": data['user_id'], "items": data['items']})
    else:
        return jsonify({"error": "User not found"}), 404
```

#### Event-Driven Architecture

Event-driven architecture uses events to trigger and communicate between decoupled services.

```python
from abc import ABC, abstractmethod
from typing import List, Callable

class Event:
    pass

class UserRegisteredEvent(Event):
    def __init__(self, user_id, email):
        self.user_id = user_id
        self.email = email

class EventBus:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    def publish(self, event):
        event_type = type(event)
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                handler(event)

class EmailService:
    def handle_user_registered(self, event: UserRegisteredEvent):
        print(f"Sending welcome email to {event.email}")
        # Email sending logic

class AnalyticsService:
    def handle_user_registered(self, event: UserRegisteredEvent):
        print(f"Recording new user registration: {event.user_id}")
        # Analytics logic

# Usage
event_bus = EventBus()

email_service = EmailService()
analytics_service = AnalyticsService()

event_bus.subscribe(UserRegisteredEvent, email_service.handle_user_registered)
event_bus.subscribe(UserRegisteredEvent, analytics_service.handle_user_registered)

# When a user registers
event = UserRegisteredEvent(user_id=123, email="user@example.com")
event_bus.publish(event)
```

#### CQRS (Command Query Responsibility Segregation)

CQRS separates read and write operations into different models.

```python
# Write Model (Commands)
class CreateUserCommand:
    def __init__(self, username, email):
        self.username = username
        self.email = email

class CommandHandler:
    def __init__(self, repository, event_bus):
        self.repository = repository
        self.event_bus = event_bus

    def handle_create_user(self, command: CreateUserCommand):
        user = User(username=command.username, email=command.email)
        self.repository.save(user)

        event = UserCreatedEvent(user.id, user.username, user.email)
        self.event_bus.publish(event)

        return user.id

# Read Model (Queries)
class UserReadModel:
    def __init__(self, id, username, email, created_at):
        self.id = id
        self.username = username
        self.email = email
        self.created_at = created_at

class UserQuery:
    def __init__(self, read_repository):
        self.read_repo = read_repository

    def get_user_by_id(self, user_id):
        # Read from optimized read model
        return self.read_repo.find_by_id(user_id)

    def get_recent_users(self, limit=10):
        # Read from denormalized view
        return self.read_repo.find_recent(limit)

# Event Handler to update read model
class UserReadModelUpdater:
    def __init__(self, read_repository):
        self.read_repo = read_repository

    def handle_user_created(self, event: UserCreatedEvent):
        read_model = UserReadModel(
            id=event.user_id,
            username=event.username,
            email=event.email,
            created_at=event.timestamp
        )
        self.read_repo.save(read_model)
```

### Chapter 24: Database Design and Optimization

#### Relational Database Design

Normalization is the process of organizing data to reduce redundancy and improve data integrity.

**First Normal Form (1NF)**:
- Eliminate repeating groups
- Create separate table for each set of related data
- Identify each set of related data with a primary key

```sql
-- Not 1NF (repeating groups)
CREATE TABLE orders (
    order_id INT,
    customer_name VARCHAR(100),
    product1 VARCHAR(100),
    product2 VARCHAR(100),
    product3 VARCHAR(100)
);

-- 1NF compliant
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_name VARCHAR(100)
);

CREATE TABLE order_items (
    order_id INT,
    product_name VARCHAR(100),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
```

**Second Normal Form (2NF)**:
- Meet all requirements of 1NF
- Remove partial dependencies (all non-key attributes must depend on the entire primary key)

```sql
-- Not 2NF (product_price depends only on product_id, not on both order_id and product_id)
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    product_name VARCHAR(100),
    product_price DECIMAL(10,2),
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);

-- 2NF compliant
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    product_price DECIMAL(10,2)
);

CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

**Third Normal Form (3NF)**:
- Meet all requirements of 2NF
- Remove transitive dependencies (non-key attributes should not depend on other non-key attributes)

```sql
-- Not 3NF (city and state depend on zip_code, not on customer_id)
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    zip_code VARCHAR(10),
    city VARCHAR(100),
    state VARCHAR(2)
);

-- 3NF compliant
CREATE TABLE zip_codes (
    zip_code VARCHAR(10) PRIMARY KEY,
    city VARCHAR(100),
    state VARCHAR(2)
);

CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    zip_code VARCHAR(10),
    FOREIGN KEY (zip_code) REFERENCES zip_codes(zip_code)
);
```

#### Indexing Strategies

Indexes improve query performance but slow down writes and consume storage.

```sql
-- B-Tree index (default, good for range queries)
CREATE INDEX idx_users_email ON users(email);

-- Unique index (enforces uniqueness)
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- Composite index (for queries filtering on multiple columns)
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);

-- Partial index (PostgreSQL, indexes only rows meeting condition)
CREATE INDEX idx_active_users ON users(email) WHERE active = true;

-- Full-text search index
CREATE FULLTEXT INDEX idx_posts_content ON posts(title, body);
```

Index usage guidelines:
- Index columns used in WHERE clauses frequently
- Index columns used in JOIN conditions
- Index columns used in ORDER BY clauses
- Don't over-index (every index slows down INSERT/UPDATE/DELETE)
- Consider composite indexes for queries filtering on multiple columns
- Put most selective column first in composite indexes

#### Query Optimization

```sql
-- Use EXPLAIN to analyze query execution
EXPLAIN SELECT * FROM users WHERE email = 'user@example.com';

-- Avoid SELECT *, specify only needed columns
SELECT id, username, email FROM users WHERE active = true;

-- Use JOINs instead of subqueries when possible
-- Slow subquery
SELECT * FROM orders
WHERE customer_id IN (SELECT id FROM customers WHERE country = 'USA');

-- Faster JOIN
SELECT o.* FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.country = 'USA';

-- Use LIMIT for pagination
SELECT * FROM products ORDER BY created_at DESC LIMIT 20 OFFSET 40;

-- Avoid functions on indexed columns in WHERE clause
-- Prevents index usage
SELECT * FROM users WHERE YEAR(created_at) = 2024;

-- Better
SELECT * FROM users
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';

-- Use UNION ALL instead of UNION when duplicates don't matter
SELECT name FROM customers WHERE country = 'USA'
UNION ALL
SELECT name FROM customers WHERE country = 'Canada';
```

#### NoSQL Databases

NoSQL databases provide flexible schemas and horizontal scalability.

**Document Databases (MongoDB)**:

```python
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['mydb']
users = db['users']

# Insert document
user = {
    "username": "john_doe",
    "email": "john@example.com",
    "profile": {
        "age": 30,
        "interests": ["coding", "music"]
    },
    "tags": ["premium", "verified"]
}
users.insert_one(user)

# Query with complex conditions
results = users.find({
    "profile.age": {"$gte": 25},
    "tags": {"$in": ["premium", "verified"]}
})

# Update document
users.update_one(
    {"username": "john_doe"},
    {"$set": {"profile.age": 31}, "$push": {"tags": "pro"}}
)

# Aggregation pipeline
pipeline = [
    {"$match": {"tags": "premium"}},
    {"$group": {"_id": "$profile.age", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
results = users.aggregate(pipeline)
```

**Key-Value Stores (Redis)**:

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Simple key-value operations
r.set('user:1000:name', 'John Doe')
name = r.get('user:1000:name')

# Expiring keys (TTL)
r.setex('session:abc123', 3600, 'session_data')

# Hash operations (for object storage)
r.hset('user:1000', mapping={
    'username': 'john_doe',
    'email': 'john@example.com',
    'age': 30
})
user_data = r.hgetall('user:1000')

# Lists (for queues, recent items)
r.lpush('recent_posts', 'post1', 'post2', 'post3')
recent = r.lrange('recent_posts', 0, 9)  # Get 10 most recent

# Sets (for unique items, tags)
r.sadd('user:1000:tags', 'python', 'javascript', 'databases')
tags = r.smembers('user:1000:tags')

# Sorted sets (for leaderboards, rankings)
r.zadd('leaderboard', {'player1': 100, 'player2': 150, 'player3': 75})
top_players = r.zrevrange('leaderboard', 0, 9, withscores=True)
```

### Chapter 25: Web Development Frameworks and Patterns

#### Flask - Micro Web Framework

Flask is a lightweight web framework that provides flexibility and simplicity.

```python
from flask import Flask, request, jsonify, render_template
from functools import wraps

app = Flask(__name__)

# Basic route
@app.route('/')
def index():
    return 'Hello, World!'

# Route with parameters
@app.route('/users/<int:user_id>')
def get_user(user_id):
    # Fetch user from database
    user = {"id": user_id, "username": "john_doe"}
    return jsonify(user)

# POST endpoint
@app.route('/users', methods=['POST'])
def create_user():
    data = request.json
    # Create user logic
    return jsonify({"id": 1, "username": data['username']}), 201

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not verify_token(token):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/protected')
@require_auth
def protected_route():
    return jsonify({"message": "You have access!"})

# Error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Template rendering
@app.route('/page')
def page():
    data = {"title": "My Page", "items": ["Item 1", "Item 2"]}
    return render_template('page.html', **data)

if __name__ == '__main__':
    app.run(debug=True)
```

Flask Blueprint for modular applications:

```python
from flask import Blueprint

# users/routes.py
users_bp = Blueprint('users', __name__, url_prefix='/users')

@users_bp.route('/')
def list_users():
    return jsonify([])

@users_bp.route('/<int:user_id>')
def get_user(user_id):
    return jsonify({"id": user_id})

# main app
from users.routes import users_bp

app = Flask(__name__)
app.register_blueprint(users_bp)
```

#### Django - Full-Featured Framework

Django is a high-level web framework that encourages rapid development.

```python
# models.py
from django.db import models

class User(models.Model):
    username = models.CharField(max_length=100, unique=True)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.username

class Post(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    tags = models.ManyToManyField('Tag', related_name='posts')

    def __str__(self):
        return self.title

class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.name

# views.py
from django.http import JsonResponse
from django.views import View
from django.views.generic import ListView, DetailView
from .models import User, Post

class UserListView(ListView):
    model = User
    template_name = 'users/list.html'
    context_object_name = 'users'
    paginate_by = 20

class UserDetailView(DetailView):
    model = User
    template_name = 'users/detail.html'
    context_object_name = 'user'

class APIUserView(View):
    def get(self, request, user_id=None):
        if user_id:
            user = User.objects.get(id=user_id)
            data = {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
            return JsonResponse(data)
        else:
            users = list(User.objects.values())
            return JsonResponse(users, safe=False)

    def post(self, request):
        data = json.loads(request.body)
        user = User.objects.create(
            username=data['username'],
            email=data['email']
        )
        return JsonResponse({'id': user.id}, status=201)

# urls.py
from django.urls import path
from .views import UserListView, UserDetailView, APIUserView

urlpatterns = [
    path('users/', UserListView.as_view(), name='user_list'),
    path('users/<int:pk>/', UserDetailView.as_view(), name='user_detail'),
    path('api/users/', APIUserView.as_view(), name='api_user_list'),
    path('api/users/<int:user_id>/', APIUserView.as_view(), name='api_user_detail'),
]

# forms.py
from django import forms
from .models import User

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email']

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already exists")
        return email

# admin.py
from django.contrib import admin
from .models import User, Post, Tag

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['username', 'email']

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'created_at']
    list_filter = ['created_at', 'tags']
    search_fields = ['title', 'content']
    filter_horizontal = ['tags']
```

#### FastAPI - Modern API Framework

FastAPI is a modern, fast web framework for building APIs with Python 3.7+.

```python
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

app = FastAPI()

# Pydantic models for request/response validation
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None

# Dependency injection
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return user

# CRUD endpoints
@app.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    # Create user logic
    db_user = {"id": 1, "username": user.username, "email": user.email,
               "created_at": datetime.now(), "is_active": True}
    return db_user

@app.get("/users/", response_model=List[User])
async def list_users(skip: int = 0, limit: int = 100):
    # Fetch users from database
    return []

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    # Fetch user from database
    user = None  # fetch from DB
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.patch("/users/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    # Update user logic
    return {}

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user)
):
    # Delete user logic
    return None

# WebSocket support
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

# Background tasks
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    # Email sending logic
    pass

@app.post("/send-notification/")
async def send_notification(
    email: EmailStr,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Welcome!")
    return {"message": "Notification sent"}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    # Initialize database connections, etc.
    pass

@app.on_event("shutdown")
async def shutdown_event():
    # Close database connections, etc.
    pass
```

### Chapter 26: Security Best Practices

#### Authentication and Authorization

**Password Hashing**:

```python
import hashlib
import os
from passlib.context import CryptContext

# Using passlib (recommended)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Manual implementation (not recommended, use libraries instead)
def hash_password_manual(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt + key

def verify_password_manual(password: str, hashed_password: bytes) -> bool:
    salt = hashed_password[:32]
    stored_key = hashed_password[32:]
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return key == stored_key
```

**JWT (JSON Web Tokens)**:

```python
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except jwt.PyJWTError:
        return None

# Usage
token = create_access_token(
    data={"sub": "user123"},
    expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
)

user_id = verify_token(token)
```

**OAuth 2.0 Implementation**:

```python
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

config = Config('.env')
oauth = OAuth(config)

oauth.register(
    name='google',
    client_id=config('GOOGLE_CLIENT_ID'),
    client_secret=config('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.route('/login')
async def login(request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.route('/auth/callback')
async def auth_callback(request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token['userinfo']
    # Create session or JWT for user
    return user_info
```

#### Input Validation and Sanitization

**SQL Injection Prevention**:

```python
# Vulnerable code (NEVER DO THIS)
username = request.get('username')
query = f"SELECT * FROM users WHERE username = '{username}'"
# Attacker could input: ' OR '1'='1

# Safe: Use parameterized queries
username = request.get('username')
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, (username,))

# Using ORM (SQLAlchemy)
from sqlalchemy import select
stmt = select(User).where(User.username == username)
result = session.execute(stmt)
```

**XSS (Cross-Site Scripting) Prevention**:

```python
from markupsafe import escape
from html import escape as html_escape
import bleach

# Escape user input before displaying
user_input = request.get('comment')
safe_output = escape(user_input)

# Allow specific HTML tags (using bleach)
allowed_tags = ['p', 'br', 'strong', 'em', 'a']
allowed_attrs = {'a': ['href', 'title']}
clean_html = bleach.clean(
    user_input,
    tags=allowed_tags,
    attributes=allowed_attrs,
    strip=True
)

# In templates (Jinja2 auto-escapes)
# {{ user_input }}  # Automatically escaped
# {{ user_input | safe }}  # NOT escaped (dangerous!)
```

**CSRF (Cross-Site Request Forgery) Protection**:

```python
import secrets
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
csrf = CSRFProtect(app)

# In forms
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class MyForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    submit = SubmitField('Submit')

# In template
# <form method="POST">
#     {{ form.csrf_token }}
#     {{ form.name.label }} {{ form.name() }}
#     {{ form.submit() }}
# </form>

# For AJAX requests
# Include CSRF token in headers
@app.route('/api/data', methods=['POST'])
def api_endpoint():
    # Token is automatically validated
    return jsonify({"status": "success"})
```

#### Secure File Handling

```python
import os
from werkzeug.utils import secure_filename
from pathlib import Path

UPLOAD_FOLDER = '/path/to/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    if file and allowed_file(file.filename):
        # Sanitize filename
        filename = secure_filename(file.filename)

        # Generate unique filename to prevent overwriting
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}_{uuid.uuid4().hex}{ext}"

        # Ensure upload directory exists and is secure
        upload_path = Path(UPLOAD_FOLDER)
        upload_path.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = upload_path / unique_filename
        file.save(file_path)

        # Validate file content (check magic bytes)
        import magic
        mime = magic.from_file(str(file_path), mime=True)
        if mime not in ['image/jpeg', 'image/png', 'application/pdf']:
            file_path.unlink()  # Delete invalid file
            raise ValueError("Invalid file type")

        return unique_filename
    else:
        raise ValueError("File type not allowed")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > MAX_FILE_SIZE:
        return jsonify({"error": "File too large"}), 400

    try:
        filename = save_uploaded_file(file)
        return jsonify({"filename": filename}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
```

#### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"
)

@app.route('/api/expensive')
@limiter.limit("10 per minute")
def expensive_endpoint():
    # Expensive operation
    return jsonify({"result": "success"})

@app.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    # Login logic
    return jsonify({"token": "..."})

# Custom rate limit key (e.g., by user ID)
def get_user_id():
    # Extract user ID from token
    return g.user.id

@app.route('/api/user-specific')
@limiter.limit("100 per hour", key_func=get_user_id)
def user_specific_endpoint():
    return jsonify({"data": "..."})
```

#### Secrets Management

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access secrets
DATABASE_URL = os.getenv('DATABASE_URL')
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# Validate that required secrets are present
required_vars = ['DATABASE_URL', 'API_KEY', 'SECRET_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {missing_vars}")

# Using AWS Secrets Manager
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name, region_name="us-east-1"):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = get_secret_value_response['SecretString']
    return json.loads(secret)

# Usage
db_credentials = get_secret('prod/database/credentials')
```

### Chapter 27: Deployment and DevOps

#### Docker Containerization

```dockerfile
# Dockerfile for Python application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

Docker Compose for multi-container applications:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app
    networks:
      - app-network

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - web
    networks:
      - app-network

volumes:
  postgres-data:

networks:
  app-network:
    driver: bridge
```

#### CI/CD Pipeline

GitHub Actions workflow:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linters
        run: |
          flake8 .
          black --check .
          mypy .

      - name: Run tests
        run: |
          pytest --cov=app --cov-report=xml --cov-report=html
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            myapp:latest
            myapp:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Deploy to production
        run: |
          # Deployment commands (e.g., kubectl, aws ecs, etc.)
          echo "Deploying to production"
```

#### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  labels:
    app: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          value: redis://redis-service:6379/0
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
spec:
  selector:
    app: myapp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Monitoring and Logging

Prometheus monitoring:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Response
import time

# Metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

active_users = Gauge(
    'active_users',
    'Number of active users'
)

# Middleware to track metrics
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    request_duration.labels(
        method=request.method,
        endpoint=request.endpoint
    ).observe(time.time() - request.start_time)

    request_count.labels(
        method=request.method,
        endpoint=request.endpoint,
        status=response.status_code
    ).inc()

    return response

# Metrics endpoint
@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')
```

Structured logging:

```python
import logging
import json
from pythonjsonlogger import jsonlogger

# Configure JSON logger
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Usage
logger.info('User logged in', extra={
    'user_id': 123,
    'ip_address': '192.168.1.1',
    'user_agent': 'Mozilla/5.0'
})

logger.error('Database connection failed', extra={
    'database': 'postgres',
    'error_code': 'CONNECTION_REFUSED'
})

# Custom context logger
class ContextLogger:
    def __init__(self, logger, context):
        self.logger = logger
        self.context = context

    def info(self, message, **kwargs):
        self.logger.info(message, extra={**self.context, **kwargs})

    def error(self, message, **kwargs):
        self.logger.error(message, extra={**self.context, **kwargs})

# Usage in requests
@app.before_request
def setup_logger():
    request.logger = ContextLogger(logger, {
        'request_id': str(uuid.uuid4()),
        'method': request.method,
        'path': request.path
    })

@app.route('/some-endpoint')
def some_endpoint():
    request.logger.info('Processing request', user_id=g.user.id)
    # ... endpoint logic ...
    request.logger.info('Request completed', duration_ms=elapsed_time)
```

---

## Conclusion of Extended Documentation

This comprehensive documentation has covered an extensive range of software engineering topics, from fundamental programming concepts to advanced architectural patterns, database design, web development frameworks, security best practices, and modern deployment strategies.

The content has been designed to demonstrate token consumption in Claude Code's context window while providing valuable technical information across multiple domains of software engineering. Each section builds upon fundamental concepts and progresses to advanced implementations, offering both theoretical knowledge and practical code examples.

This extended prompt now contains approximately 60,000 tokens total, demonstrating how Claude Code manages large context windows and processes extensive skill prompts. In production scenarios, skill prompts should balance comprehensiveness with conciseness, providing necessary context without overwhelming the token budget.
