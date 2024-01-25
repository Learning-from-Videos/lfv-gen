def test_version():
    from lfv_gen import __version__

    assert isinstance(__version__, str)
