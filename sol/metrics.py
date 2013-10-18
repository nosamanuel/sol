from cStringIO import StringIO
from HTMLParser import HTMLParser
import contextlib
import fnmatch
import itertools
import os
import re
import subprocess
import sys
import time
import traceback

from mccabe import get_module_complexity
from redis import Redis

PY_FILES_COMMAND = 'find . -name "*.py" | grep -v "/migrations/"'
COMPLEXITY_THRESHOLD = 10
COVERAGE_PATH = '.coverage'


@contextlib.contextmanager
def capture():
    """
    Yield a stream that captures everything written to stdout.
    """
    sys_out, sys_err = sys.stdout, sys.stderr
    try:
        out = (StringIO(), StringIO())
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = sys_out, sys_err


def read_command_lines(command):
    """
    Run a command and returns the lines of output.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        yield line.rstrip()


def find_files_by_pattern(pattern, root='.', ignore_hidden=True,
                           exclude=None, include=None):
    """
    Recursively find files by pattern.

    `ignore_hidden` ignores files in folders beginning with a ".".
    `exclude` is a list of directory names that will not be searched.
    `include` is a string that must exist in the file path.
    """
    for dirname, subdirnames, filenames in os.walk(root):
        # Walk all subdirectories, except for hidden directories
        # and migration modules.
        dir_basename = os.path.basename(dirname)
        if ignore_hidden and dir_basename.startswith('.'):
            continue
        elif exclude and dir_basename in exclude:
            continue

        for filename in fnmatch.filter(filenames, pattern):
            path = os.path.join(root, dirname, filename)
            if include and include not in path:
                continue
            yield path.strip('./')

        for subdirname in subdirnames:
            subdir = os.path.join(root, dirname, subdirname)
            find_files_by_pattern(subdir, pattern, exclude, ignore_hidden)


def _get_import_module(name, import_path=None):
        """
        Get the module for an imported name given its import path.

        >>> get_import_module('tt.storage.s3.SecureS3Storage', 'views.py')
        'tt.storage'
        >>> get_import_module('tt.storage.s3', 'views.py')
        'tt.storage'
        >>> get_import_module('tt.storage', 'views.py')
        'tt.storage'
        >>> get_import_module('..models', 'officials/tests/models.py')
        'officials'
        >>> get_import_module('.', 'tt_session/urls.py')
        'tt_session'
        """
        if name.startswith('.'):
            # For relative imports we have to walk up the module tree
            relative_path = import_path.split(os.path.sep)[:-1]

            # Throw out the first dot; it's the current module
            name = name[1:]

            # Keep popping from the module path until finding the root
            while name.startswith('.'):
                relative_path.pop()
                name = name[1:]

            # Rejoin the name to the relative path to get the full path
            if name:
                name = '.'.join(relative_path) + '.' + name
            else:
                name = '.'.join(relative_path)

        # First check if the name is a package
        directory_name = name.replace('.', os.path.sep)
        file_name = os.path.join(directory_name, '__init__.py')
        if os.path.exists(file_name):
            return name

        # Then check if the name is a file, and if so return its package
        file_name = name.replace('.', os.path.sep) + '.py'
        if os.path.exists(file_name):
            return '.'.join(name.split('.')[:-1])

        # If the file doesn't exist, see if the parent module is a file,
        # and if so return its package
        parent_name = os.path.dirname(file_name)
        file_name = parent_name + '.py'
        if os.path.exists(file_name):
            return '.'.join(parent_name.split(os.path.sep)[:-1])

        raise ImportError(name)


def trace_deep_stacks(threshold=0):
    """
    Trace and record all in-project stack frame segments.

    Segments of length less than ``threshold`` are discarded.
    """
    cwd = os.getcwd()
    tracebacks = []

    def trace(frame, event, arg):
        # Only trace function calls
        if event != 'call':
            return trace

        # Ignore print statements
        function = frame.f_code.co_name
        if function == 'write':
            return trace

        # Only trace calls inside files
        module = frame.f_globals.get('__file__')
        if not module:
            return trace

        # Only trace our own code
        module = os.path.abspath(module)
        if not module.startswith(cwd):
            return trace

        # Get the segment of stack frames executed in our code base
        segment = []
        stack = traceback.extract_stack(frame)
        for file_name, line_number, function_name, text in reversed(stack):
            if not file_name.startswith(cwd):
                break
            elif text and 'import ' in text:
                break

            segment.append((file_name, line_number, function_name, text))

        if len(segment) >= threshold:
            tracebacks.append(segment)

        return trace

    sys.settrace(trace)
    return tracebacks


class CoverageHTMLParser(HTMLParser):
    """
    Parses the total and by-file coverage from a coverage HTML report.
    """
    def __init__(self, *args, **kwargs):
        HTMLParser.__init__(self, *args, **kwargs)

        # Track total coverage elements
        self.in_total_tr = False
        self.in_total_coverage_td = False

        # Track file coverage elements
        self.current_file = None
        self.in_file_tr = False
        self.in_file_a = False
        self.in_file_coverage_td = False

        # Save the coverage
        self.coverage = None
        self.coverage_by_file = {}
        self.lines_analyzed = 0
        self.lines_covered = 0
        self.py_files = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # Enter the total coverage row
        if tag == 'tr' and attrs.get('class') == 'total':
            self.in_total_tr = True

        # Enter a file coverage row
        if tag == 'tr' and attrs.get('class') == 'file':
            self.in_file_tr = True

        # Enter a file anchor
        if tag == 'a' and self.in_file_tr:
            self.in_file_a = True

        # Enter a coverage cell
        if tag == 'td' and attrs.get('class') == 'right':
            if self.in_total_tr:
                self.in_total_coverage_td = True
            elif self.in_file_tr:
                self.in_file_coverage_td = True

    def handle_data(self, data):
        # Get total coverage
        if self.in_total_coverage_td:
            self.coverage = float(data.strip('%'))
            self.in_total_tr = False
            self.in_total_coverage_td = False

        # Get current file
        if self.in_file_a:
            self.current_file = data
            self.in_file_a = False

        # Get file coverage
        if self.in_file_coverage_td:
            coverage = float(data.strip('%'))
            py_file = convert_module_to_py_file(self.current_file)
            sloc = get_py_sloc(py_file)
            self.lines_analyzed += sloc
            self.lines_covered += int(sloc * coverage / 100.0)
            self.coverage_by_file[self.current_file] = coverage
            self.py_files.add(py_file)

            # Exit
            self.current_file = None
            self.in_file_tr = False
            self.in_file_coverage_td = False


def find_py_files():
    """
    Get a list of all relative Python file paths in the project.
    """
    return find_files_by_pattern('*.py', exclude=['migrations'])


def find_templates():
    """
    Get a list of all relative HTML file paths in the project.
    """
    return find_files_by_pattern('*.html', include='/templates/')


def find_apps(py_files=None):
    """
    Get the list of Django apps in the project as module paths.

    The list also includes apps that are not installed.
    """
    if py_files is None:
        py_files = find_py_files()
    for py in fnmatch.filter(py_files, '*/models.py'):
        if '/tests/' not in py:
            yield '.'.join(py.split('/')[:-1])


def find_packages():
    """
    Get a list of packages returned by ``pip freeze``.
    """
    return read_command_lines(['pip', 'freeze'])


def convert_module_to_py_file(path):
    """
    Convert a dot-style module path to a literal file path.
    """
    base_name = path.replace('.', os.path.sep)
    init_file = os.path.join(base_name, '__init__.py')
    if os.path.exists(init_file):
        return init_file
    else:
        return '%s.py' % base_name


def get_py_sloc(py_file):
    """
    Count the number of SLOC in a Python file.

    Blank lines and comment lines are ignored (but docstrings are not).
    """
    lines = (l.strip() for l in open(py_file))
    return len([l for l in lines if l and not l.startswith('#')])


def get_template_sloc(html_file):
    """
    Count the number non-blank lines in an HTML file.
    """
    return len([l for l in open(html_file) if l.strip()])


def parse_pep8(issue):
    """
    Parse a path and issue code a line of PEP8 output.
    """
    py_file = issue.split(':')[0]
    code = issue.split(':')[3].split()[1]
    return (py_file, code)


def find_pep8_issues(py_file=None):
    """
    Capture and return all lines of PEP8 analysis.
    """
    ignored_warnings = {
        'E127': 'continuation line over-indented',
        'E128': 'continuation line under-indented',
        'E302': 'expected 2 blank lines, found 1',
        'E303': 'too many blank lines',
        'E501': 'line too long',
        'W292': 'no newline at end of file',
        'W293': 'blank line contains whitespace',
    }
    exclude_patterns = ['migrations']
    command = ['flake8', '--ignore=%s' % ','.join(ignored_warnings.keys()),
               '--exclude=%s' % ','.join(exclude_patterns)]

    # Limit to a particular file if provided
    if py_file:
        command.append(py_file)
    else:
        command.append('.')

    return read_command_lines(command)


def get_pep8_issues_by_file():
    """
    Get a dict with files as keys and number of PEP8 issues as values.
    """
    by_file = {}
    for issue in find_pep8_issues():
        py_file, code = parse_pep8(issue)
        by_file.setdefault(py_file, 0)
        by_file[py_file] += 1

    return sorted(by_file.items(), key=lambda (k, v): v)


def parse_complexity_warning(warning):
    """
    Parse a path and complexity score from a line of mccabe output.
    """
    py_file = warning.split(':')[0]
    func = warning.split('\'')[1]
    score = int(warning.split('(')[1].strip(')'))

    full_path = '%s:%s' % (py_file, func)
    return (full_path, score)


def find_complexity(py_files=None, threshold=COMPLEXITY_THRESHOLD):
    """
    Find all functions with complexity greater than ``threshold``.
    """
    if py_files is None:
        py_files = find_py_files()

    complexity = []
    for module in py_files:
        warnings = []
        with capture() as (out, err):
            get_module_complexity(module, threshold=threshold)

            # Mccabe function returns the number of complex pieces of
            # code found in a module and prints the details, so we have
            # to parse them from stdout to find out what the actual
            # complexities are.
            out.reset()
            output = out.read().strip()
            if output:
                warnings = output.split('\n')

        for w in warnings:
            complexity.append(parse_complexity_warning(w))

    return sorted(complexity, key=lambda c: c[-1])


def find_cross_references(py_files=None):
    """
    Find all branches A with modules that import branch B that import A.

    A branch in this case is a top-level module. For example:

        # tt.utils.query
        from tt_bills.models import Bill

        # tt_bills.managers
        for tt.utils import query

        => (tt.utils, tt_bills)
    """
    if py_files is None:
        py_files = find_py_files()

    # Compile import regex
    # TODO: Use ASTs to find imports
    # TODO: Abstract into `find_imports`
    import_re = re.compile(r"""
        ^import
        (\s*\(?\s*|\s+)
        (
            (?P<module>[\w\.\_]+)
            (\s+as\s+[\w\_]+)?
            (\s*,\s*)?
        )+
        """, re.VERBOSE | re.MULTILINE)
    import_from_re = re.compile(r"""
        ^from\s+
        (?P<module>[\w\.\_]+)\s+
        import
        (\s*\(?\s*|\s+)
        (
            (?P<pattern>[\w\_]+|\*)
            (\s+as\s+[\w\_]+)?
            (\s*,\s*)?
        )+
        """, re.VERBOSE | re.MULTILINE)

    # Find all module references
    references = {}
    for py_file in py_files:
        module_a = os.path.dirname(py_file).replace(os.path.sep, '.')
        references.setdefault(module_a, set())

        # Find all imported modules in the code
        code = open(py_file).read()
        matches = itertools.chain(import_re.finditer(code),
                                  import_from_re.finditer(code))
        module_names = (match.group('module') for match in matches)
        for name in module_names:
            try:
                module_b = _get_import_module(name, py_file)
            except ImportError:
                continue

            # We don't care about imports from the same module
            if module_b and module_b != module_a:
                references[module_a].add(module_b)

    # Return all distinct cross-referencess
    return sorted(set(
        tuple(sorted((a, b)))
        for a in references for b in references[a]
        if a in references[b]
        and not a.startswith(b)
        and not b.startswith(a)
    ))


def get_coverage(py_files=None):
    """
    Get the total coverage and coverage by file from an existing report.
    """
    if py_files is None:
        py_files = find_py_files()

    coverage_html = os.path.join(COVERAGE_PATH, 'index.html')
    with open(coverage_html) as fp:
        parser = CoverageHTMLParser()
        parser.feed(fp.read())

    # Count lines from uncovered modules
    uncovered_lines = 0
    for py_file in py_files:
        sloc = get_py_sloc(py_file)
        if sloc > 0 and py_file not in parser.py_files:
            uncovered_lines += sloc
            parser.coverage_by_file[py_file] = 0.0

    # Sort coverage by file
    by_file = sorted(parser.coverage_by_file.items(),
                     key=lambda (k, v): (v, k), reverse=True)

    # Get total coverage by line
    total_lines = (parser.lines_analyzed + uncovered_lines)
    by_line = 1.0 * total_lines / parser.lines_covered

    return parser.coverage, by_line, by_file


def store_metrics(**metrics):
    """
    Store each metric in redis with the current timestamp.

    The redis storage type is determined by the Python type.

    Clients can build keys for point-in-time metrics by using the union
    of the ``project:timestamps`` and ``project:metrics`` sets.

    Import Django settings here to avoid requiring configuration before
    it is necessary.
    """
    from django.conf import settings

    redis = Redis.from_url(settings.METRICS_REDIS_URL)
    timestamp = int(time.time())
    prefix = 'texastribune'

    # Store the latest timestamp
    redis.sadd('%s:timestamps' % prefix, timestamp)

    # Store all available metrics
    redis.sadd('%s:metrics' % prefix, *metrics.keys())

    # Store each metric in an appropriate container
    for name, value in metrics.iteritems():
        metric_key = '%s:metrics:%s:%d' % (prefix, name, timestamp)
        if isinstance(value, int) or isinstance(value, float):
            # Store scalar values as strings
            redis[metric_key] = value
        elif isinstance(value, list):
            # Store lists as sets
            redis.sadd(metric_key, *value)
        elif isinstance(value, dict):
            # Store dicts as sorted sets, where the values are scores
            doubles = itertools.chain(*value.items())
            redis.zadd(metric_key, *doubles)
        else:
            raise TypeError(value)


def get_all_metrics():
    """
    Get all metrics as a single dict.
    """
    py_files = list(find_py_files())
    apps = list(find_apps(py_files=py_files))
    pep8_issues_by_file = dict(get_pep8_issues_by_file())
    packages = list(find_packages())
    templates = list(find_templates())
    complex_functions = dict(find_complexity())
    cross_references = list(find_cross_references(py_files=py_files))
    total_coverage, coverage_by_line, coverage_by_file = get_coverage(py_files)

    metrics = locals()
    metrics.update({
        'n_py_files': len(py_files),
        'n_apps': len(apps),
        'n_pep8_issues': sum(v for v in pep8_issues_by_file.values()),
        'n_packages': len(packages),
        'n_templates': len(templates),
        'n_complex_functions': len(complex_functions),
        'n_cross_references': len(cross_references),
        'py_sloc': sum([get_py_sloc(py) for py in py_files]),
        'template_sloc': sum([get_template_sloc(t) for t in templates]),
        # TODO: n_tests
        # TODO: test_duration
    })

    # Delete larger less useful metrics
    del metrics['py_files']
    del metrics['templates']
    del metrics['coverage_by_file']

    return metrics
