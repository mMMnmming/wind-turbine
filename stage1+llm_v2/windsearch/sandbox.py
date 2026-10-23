"""Exception shared with the trusted, killable Stage2 worker."""
class CandidateDeadlineExceeded(TimeoutError):
    pass
