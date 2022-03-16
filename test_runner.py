import re
import __main__
import textwrap
from functools import wraps
import sys
import os

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)
except ModuleNotFoundError:
    CSI = '\033['

    def code_to_chars(code):
        return CSI + str(code) + 'm'

    class AnsiCodes(object):
        def __init__(self):
            for name in dir(self):
                if not name.startswith('_'):
                    value = getattr(self, name)
                    setattr(self, name, code_to_chars(value))

    class AnsiFore(AnsiCodes):
        BLACK = 30
        RED = 31
        GREEN = 32
        YELLOW = 33
        BLUE = 34
        MAGENTA = 35
        CYAN = 36
        WHITE = 37
        RESET = 39

    class AnsiBack(AnsiCodes):
        BLACK = 40
        RED = 41
        GREEN = 42
        YELLOW = 43
        BLUE = 44
        MAGENTA = 45
        CYAN = 46
        WHITE = 47
        RESET = 49

    class AnsiStyle(AnsiCodes):
        BRIGHT = 1
        DIM = 2
        NORMAL = 22
        RESET_ALL = 0

    Fore = AnsiFore()
    Back = AnsiBack()
    Style = AnsiStyle()


class SuppressPrint:
    """
    A context manager that will suppress print if the constructor arg is true.
    """

    def __init__(self, suppress_print=False):
        self.__suppress_print = suppress_print

    def __enter__(self):
        if self.__suppress_print:
            self._stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__suppress_print:
            sys.stdout.close()
            sys.stdout = self._stdout


class TestRunner:
    """
    A lite unittest utility.
    """
    __skip_tests = []
    __points_matrix = {}

    def __init__(self, *, verbose=True, fail_fast=False, suppress_print=False):
        self.__verbose = verbose
        self.__fail_fast = fail_fast
        self.__suppress_print = suppress_print
        self.__score_card = {test_name: 0 for test_name in self.__points_matrix.keys()}

    @staticmethod
    def match_test_name_pattern(name, re_pattern, flags=0):
        pattern = re.compile(re_pattern, flags=flags)
        if pattern.match(name):
            return True
        else:
            return False

    @classmethod
    def skip(cls, fn):
        cls.__skip_tests.append(fn.__name__)

        @wraps(fn)
        def wrapper():
            fn()

        return wrapper

    @classmethod
    def points(cls, points):
        def decorator(fn):
            cls.__points_matrix[fn.__name__] = points

            @wraps(fn)
            def wrapper():
                fn()
            return wrapper
        return decorator

    def run_tests(self, *test_names, ensure_test=True):
        """

        :param test_names: List[str] - list of test names to run.
        :param ensure_test: bool - If True only run tests that start with test_
        :return: None
        """
        if ensure_test:
            test_names = list(filter(lambda name: self.match_test_name_pattern(name, r"^test\_"), test_names))
        self.isolate_tests = test_names
        self.run()

    def __run_test(self, test_func):
        with SuppressPrint(suppress_print=self.__suppress_print):
            test_func()

    def run(self):
        """
        Runs all the tests in the __main__ module.
        """

        success_count = 0
        fail_count = 0
        failed_tests = []

        all_tests = getattr(self,
                            "isolate_tests",
                            list(filter(lambda name: self.match_test_name_pattern(name, r"^test\_"),
                                 __main__.__dict__.keys())))

        active_tests = [test_name for test_name in all_tests if test_name not in self.__skip_tests]

        col_size = max([len(name) for name in active_tests]) + len(str(len(active_tests))) + 5

        text_wrapper = textwrap.TextWrapper(width=80, initial_indent='\t\t', subsequent_indent='\t\t\t')

        print("\nRUNNING TESTS...", end="\n\n")

        try:
            if not self.__verbose:
                print("\t", end="")
            for i, test_name in enumerate(active_tests):
                verbose_output = ""
                if self.__verbose:
                    verbose_output = f"\t{i+1}. {test_name}: "
                test = getattr(__main__, test_name)
                try:
                    self.__run_test(test)
                except AssertionError as e:
                    fail_count += 1
                    failed_tests.append((test_name, None))
                    output = f"{Fore.RED}F{Fore.RESET}"
                    if self.__verbose:
                        output = f'{verbose_output: <{col_size}}{Fore.RED}Failed\n{Fore.RESET}'
                    print(output, end="")
                    print(f"{Fore.YELLOW}{text_wrapper.fill(f'Reason: {e}')}{Fore.RESET}") \
                        if self.__verbose else print(end="")
                    if self.__fail_fast:
                        break
                except Exception as e:
                    exc_type = type(e).__name__
                    fail_count += 1
                    failed_tests.append((test_name, exc_type))
                    output = f"{Fore.RED}E{Fore.RESET}"
                    if self.__verbose:
                        output = f'{verbose_output: <{col_size}}{Fore.RED}Failed - {exc_type}\n{Fore.RESET}'
                    print(output, end="")
                    print(f"{Fore.YELLOW}{text_wrapper.fill(f'Reason: {e}')}{Fore.RESET}") \
                        if self.__verbose else print(end="")
                    if self.__fail_fast:
                        break
                else:
                    output = '.'
                    if self.__verbose:
                        output = f'{verbose_output: <{col_size}}{Fore.GREEN}Passed{Fore.RESET}\n{Fore.RESET}'
                    print(output, end="")
                    self.__score_card[test_name] = self.__points_matrix[test_name]
                    success_count += 1
        except KeyboardInterrupt as e:
            print(f"{Fore.RED}Failed{Fore.RESET}")
            print(f"\t{Fore.YELLOW}Reason: {type(e)}{Fore.RESET}")

        if not self.__verbose:
            print()
            print("\nFAILED TESTS:")
            if failed_tests:
                for test, exc_type in failed_tests:
                    output = f"\t{Fore.RED}{test}{f' - {exc_type} Raised' if exc_type else ''}{Fore.RESET}"
                    print(output)
            else:
                print("\tNone")

        print("\nSTATISTICS:")

        print(f"\n\t{Fore.CYAN}Checked {success_count + fail_count} out of {len(active_tests)} active tests."
              f"{Fore.RESET}")
        print(f"\t{Fore.CYAN}Skipped {len(self.__skip_tests)} tests.{Fore.RESET}")

        print(f"\n\t{Fore.CYAN}Passed {success_count} out of {success_count + fail_count} "
              f"({success_count / (success_count + fail_count):.00%}) tests.{Fore.RESET}")

        total_points = sum(self.__points_matrix.values())
        pass_points = sum(self.__score_card.values())

        print(f"\n\t{Fore.CYAN}Score: {pass_points} / {total_points} ({pass_points/total_points:.00%}){Fore.RESET}")

        print("\nCOMPLETE")
