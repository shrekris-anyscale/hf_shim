"""This file contains global state during execution.

Any global counters should be stored here. Their state is shared across
asynchronous tasks.
"""

####################
# Global variables #
####################

# Do not access these directly. Use accessor methods instead.

_request_id = 0
_batch_id = 0


####################
# Accessor methods #
####################

# Import these and use them to modify or access global variables.


def get_and_increment_request_id() -> int:
    global _request_id
    _request_id += 1
    return _request_id


def reset_batch_id() -> None:
    global _batch_id
    _batch_id = 0


def get_and_increment_batch_id() -> int:
    global _batch_id
    _batch_id += 1
    return _batch_id
