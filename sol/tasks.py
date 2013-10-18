import json

from invoke import task

from .metrics import COMPLEXITY_THRESHOLD
from .metrics import (get_all_metrics, get_coverage, get_pep8_issues_by_file,
    find_apps, find_complexity, find_cross_references, find_packages,
    find_pep8_issues, store_metrics)


@task(default=True)
def all(store=False):
    """
    Prints all metrics as indented JSON.

    With the ``--store`` flag, metrics will be stored in Redis.
    """
    metrics = get_all_metrics()
    print json.dumps(metrics, indent=2)

    if store:
        store_metrics(**metrics)


@task
def coverage():
    """Prints coverage by file with totals"""
    total, by_line, by_file = get_coverage()
    for c in by_file:
        print c

    print 'Total: %.2f%%' % total
    print 'By line: %.2f%%' % by_line


@task
def pep8(py_file=None):
    """
    Prints the number of PEP8 issues by file.

    Pass in ``py_file`` to show all the PEP8 warnings in a single file.
    """
    if py_file:
        for warning in find_pep8_issues(py_file=py_file):
            print warning
    else:
        for i in get_pep8_issues_by_file():
            print i


@task
def apps():
    """Prints all Django apps found in the project"""
    for app in find_apps():
        print app


@task
def packages():
    """Prints all packages required by the project"""
    for package in find_packages():
        print package


@task
def complexity(threshold=COMPLEXITY_THRESHOLD):
    """Prints all functions with complexity ``threshold`` or greater"""
    threshold = int(threshold)
    for c in find_complexity(threshold=threshold):
        print c


@task
def cross_references():
    """Prints all sets of modules that import each other"""
    for x in find_cross_references():
        print x
