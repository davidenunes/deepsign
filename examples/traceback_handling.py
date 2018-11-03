import traceback


def fn1():
    def fn2():
        try:
            raise TypeError()
        except Exception as e2:
            raise e2
    try:
        fn2()
    except Exception as e:
        return e


def main():
    e = fn1()

    tb = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__, limit=-2))
    print(tb)

main()
