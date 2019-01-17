from collections import Callable


class Spec:
    """ Specs are to be used with Layer class because otherwise we have no __spec__ vars


    There is also a library called traits
    https://pypi.org/project/traits/

    there should be more than one lib to do this

    This makes a good post about api design
    https://stackoverflow.com/questions/40584140/maximum-recursion-depth-exceeded-when-using-a-class-descriptor-with-get-and-set
    """

    def __init__(self, types=object, required=False, forward=False):
        self.types = types
        self.required = required
        self.forward = forward
        self.cls = None

    def __get__(self, instance, cls):
        print('Retrieving', self.name)
        print('obj type ', cls)
        # triggered when we retrieve the descriptor from the class directly
        if instance is None:
            print("none instance for ", self.name)
            return self
        try:
            return instance.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name)

    def __set__(self, instance, value):
        if not isinstance(value, self.types):
            raise TypeError("{name} has type {t}: received value of type {wrong_t}".format(name=self.name,
                                                                                           t=self.types,
                                                                                           wrong_t=type(value)))

        print('Updating', self.name)
        instance.__dict__[self.name] = value

    # this let us know the name of the attribute the descriptor is assigned to
    # owner is the class object where the descriptor is defined
    # this is called when a new descriptor is created
    def __set_name__(self, cls, name):
        # print("the name of the attribute is ", name)
        # print(cls)
        self.name = name
        self.cls = cls
        if not hasattr(cls, "__spec__"):
            cls.__spec__ = set()

            def _spec(): return {s.name: s for s in cls.__spec__}

            cls._spec = _spec
        cls.__spec__.add(self)

    def __delete__(self, instance):
        del instance.__dict__[self.name]


class A:
    spec1 = Spec((int, str), required=True)
    spec2 = Spec((str, float))


class B(A):
    spec3 = Spec()


class C(A, B):
    reset = Spec(Callable)

    # cannot do this with functions but, that said functions
    def reset(self):
        print("resetting some shit")


a = A()
b = B()

a.spec1 = 3
a.spec1 = "hello"
# a.spec2 = 3

print(a.__spec__)
print(A.__spec__)

print("B spec: ", B.__spec__)
print(B.spec3)
print(B.spec3.cls)
print(A.spec2.cls)
print(B.spec1.cls)

print(B._spec())
s = B._spec()
print("spec6" in s)
# print(getattr(A, "spec1"))
# print(getattr(a, "spec1"))
# a.spec1_2 = 2

# print(a.spec1_2)
