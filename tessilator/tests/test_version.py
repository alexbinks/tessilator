def test_version():
    from tessilator import __version__
    assert __version__ is not None, "Version should not be None"
    assert isinstance(__version__, str), "Version should be a string"
    assert len(__version__) > 0, "Version string should not be empty"