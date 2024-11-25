def hailstone(n: int) -> None:
    """Prints out the hailstone sequence starting at integer n. Assumes n > 0

    Args:
        n (int): Specifies the starting number of the hailstone sequence
    """
    assert n > 0
    while n != 1:
        print(n)
        if n % 2 == 0:
            n //= 2
        else:
            n = 3*n + 1
    print(n)
