# TestRunner

<hr>
NOTICE: This module is only to help you write solutions offline. The tests included as well as the assigned points 
may or may not reflect what's actually being tested in Hackerrank or otherwise. Use this at your own discretion. 
<hr>

The test_runner module was designed to help you work the Hackerrank test cases offline. It is very much like a lite
version of the builtin unittest library.

It's very simple to use:

```python
from test_runner import TestRunner

def test_my_func():
    assert True
    
def test_your_func():
    assert False, "Finish the function"

if __name__ == '__main__':
    runner = TestRunner(verbose=True, fail_fast=True, suppress_print=True)
    runner.run()
```
produces output to console:

```cmd
RUNNING TESTS...

	1. test_my_func:                             Passed
	2. test_your_func:                           Failed
        Reason: Finish the function 

STATISTICS:

	Checked 2 out of 2 active tests.
	Skipped 0 tests.

	Passed 1 out of 2 (50%) tests.

	Score: 1 / 2 (50%)

COMPLETE
```

At runtime the TestRunner will search through the module and register all functions that begin with test. 


#### Arguments
1. verbose - will condense or expand the output. 
2. fail_fast - when true will halt further execution of test cases upon the first fail either due to Assertion Error 
or other exception.
3. suppress_print - useful where you want to keep the console clean of extraneous prints and just check if you passed 
a test case or not. 

## Isolate Tests
Sometimes you want to run just a few test functions. This can be done in one of two ways. 

1. Explicitly run only selected tests.

```python
    runner = TestRunner()
    isolated_tests = ("test_my_func", "test_your_func")
    runner.run_tests(*isolated_tests)
```

2. Skip tests

```python
from test_runner import TestRunner

@TestRunner.skip()
def test_my_func():
    assert True
```


## Scoring

Tests may also include a score

```python
from test_runner import TestRunner

@TestRunner.point(10)
def test_my_func():
    assert True
```