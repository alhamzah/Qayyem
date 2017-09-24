import sys


class EvaluationError(Exception):
    """Exception to be raised if there is an error during evaluation."""
    pass

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a carlae
                      expression
    """
    a = source.split('\n')
    a = [s.partition(';')[0] for s in a]    #remove comments
    next1 = [i for word in a for i in word.split()] #remove white-spaces

    result = []
    def sep(s, result=result, s_char={'(', ')'}):   #Seperate into lists
        if not s or s[0] == ';': return
        for i in range(len(s)):
            if s[i] in s_char:
                sep(s[:i])
                result.append(s[i])
                return sep(s[i+1:])
        result.append(s)

    for word in next1:
        sep(word)
    return result


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    new_tokens = []
    for word in tokens:
        try:
            if int(float(word)) == float(word):     #Handling numbers
                new_tokens.append(int(word))
            else:
                new_tokens.append(float(word))
        except ValueError:
            new_tokens.append(word)

    if new_tokens.count(')') != new_tokens.count('('):
        raise SyntaxError

    def find_mate(tokens):
        '''Find the other paranthesis'''
        par_count = 1
        for i in range(len(tokens)):
            if tokens[i] == '(': par_count += 1
            elif tokens[i] == ')': par_count -= 1
            if par_count == 0: return i+1
        raise SyntaxError

    def add_token(tokens):
        for i in range(len(tokens)):
            if tokens[i] == '(':
                result = []
                result.extend(tokens[:i])
                e = find_mate(tokens[i+1:]) + i
                result.append(add_token(tokens[i+1:e]))
                return result + add_token(tokens[e+1:])
        return tokens
    return add_token(new_tokens)[0]

class Trie():
    '''Environment representation.

    Attributes:
        name (str): environment's name
        root (Trie): Trie's root environment
        parent (Trie): Trie's parent environment
        children (dict): Trie's children environment
        keys (dict): variable's defined in the environment

    Methods:
        insert_key(var, value): define variable var in the environment with value 'value'
        insert_env(env): create a new child environment env
        insert_empty_env(): create an empty child environment with random name
        remove_env(env): remove a child environment
        find_key(key): return the key value if it exist in the environment, otherwise look at the parents
        find_env(env): return the Trie object named env
        print_trie(): print the Trie starting from the root
        '''

    def __init__(self, name):
        self.name, self.root, self.parent, self.children, self.keys = name, self, None, dict(), dict()

    def insert_key(self, var, value):
        self.keys[var] = value

    def insert_env(self, env):
        child = Trie(str(env))
        child.parent, child.root, self.children[env] = self, self.root, child

    def insert_empty_env(self):
        self.insert_env(id(self))
        wenv = self.find_env(id(self))
        self.insert_env(id(wenv))
        wenv = self.find_env(id(wenv))
        self.remove_env(id(self))
        return wenv

    def remove_env(self, env):
        parent = self.find_env(env).parent
        parent.children.pop(env)

    def find_env(self, env):
        if not env: return envs
        if env in self.children: return self.children[env]
        for t in self.children.values():
            result = t.find_env(env)
            if result: return result

    def find_key(self, key, env=False):
        if key in self.keys:
            if not env: return self.keys[key]
            return self.keys[key], self
        if self == self.root:
            raise KeyError ('المفتاح غير موجود في البيئة الحالية')
        return self.parent.find_key(key, env)

    def print_trie(self, v=False, prefix= '      ', h=False):
        print((len(prefix)-6)*' '+'|'+'-'*4+' '+self.name)
        for key in self.keys:
            if v == True:
                value = self.find_key(key)
                if callable(value):
                    print(prefix+'  |-'+' '+str(key)+'   <function>')
                else:
                    print(prefix+'  |-'+' '+str(key)+' = '+str(value))
            else: print(prefix+'  |-'+' '+str(key))
        if h: return
        for env in self.children.values():
            env.print_trie(v, '      ' + prefix)


def define(args):
    env, value = args[2], evaluate(args[1], args[-1])
    env.insert_key(args[0], value); 
    return value

def def_f(argList):
    '''Define Carlae's functions'''
    para, expression, env = argList[0], argList[1], argList[2]
    # print(argList, '-----------')
    def evaluate_f(args):
        # if len(args) != len(para): raise TypeError ('Type error: args: ', args, 'para: ', para)
        parent = env
        wenv = parent.insert_empty_env()
        for i in range(len(args)):
            wenv.insert_key(para[i], args[i])
        return evaluate(expression, wenv)
    return evaluate_f

def c_if(args):
    if evaluate(args[0], args[-1]) == '#ص': return evaluate(args[1], args[-1])
    elif evaluate(args[0], args[-1]) == '#خ': return evaluate(args[2], args[-1])
    raise CondError ('Invalid condition value')

def def_c(r1, r2=None):
    '''Define carlae's comparison operators'''
    compare = lambda a,b: (a > b) - (a < b)     #returns 1 if a > b, 0 if a == b, -1 if a < b
    def c(args):
        for i in range(len(args)-1):
            if compare(args[i], args[i+1]) != r1:
                if r2:
                    if compare(args[i], args[i+1]) != r2: return '#خ'
                else: return '#خ'
        return '#ص'
    return c

def c_and(args):
    for i in range(len(args)-1):
        if evaluate(args[i], args[-1]) != '#ص': return '#خ'
    return '#ص'

def c_or(args):
    for i in range(len(args)-1):
        if evaluate(args[i], args[-1]) == '#ص': return '#ص'
    return '#خ'


class LinkedList():
    '''Linked-lists representation.

    Attributes:
        elt: value stored at the instance. 'NIL' if it is the empty list.
        next: next object in the list. 'None' if it is the last element in the list or the empty list.

    Methods:
        length: return the length of the linked list starting from the instance.
        copy: return a copy of the instace.
        last_elt: reutrn the last elemeent in the list.
        cdr: return the list containing all but the first element in the list.
        elt_at_index(i): return the LinkedList object of index i of the list.
    '''
    def __init__(self, elt, nelt=None):
        self.elt = elt
        self.next = nelt

    def __repr__(self):
        if self.get_elt == 'NIL': return '[]'
        def helper(self, start=True):
            s = str(self.get_elt())
            if not self.get_nelt():
                return'['+s+']' if start else ', '+s+'>'
            nelt = self.get_nelt()
            return '<'+s+helper(nelt, False) if start else ', '+s+helper(nelt, False)
        return helper(self)

    def get_elt(self):
        return self.elt

    def get_nelt(self):
        return self.next

    def set_nelt(self, nelt):
        self.next = nelt

    def length(self):
        if self.get_elt() == 'NIL': return 0
        return 1 if not self.get_nelt() else 1 + self.get_nelt().length()

    def copy(self):
        return LinkedList(self.get_elt()) if not self.get_nelt() else LinkedList(self.elt, self.get_nelt().copy())

    def last_elt(self):
        return self if not self.get_nelt() else self.get_nelt().last_elt()

    def cdr(self):
        return self.get_nelt()

    def elt_at_index(self, index, value=True):
        if self.length()-1 < index: raise EvaluationError
        alist, counter = self, 0
        while counter != index:
            alist = alist.cdr()
            counter += 1
        return alist

def build_list(args):
    '''Create a LinkedList object from a given python array.'''
    if args[0] == []: return LinkedList('NIL')
    start, env = LinkedList(args[0]), args[-1]
    current = start

    for i in range(1,len(args)): 
        nelt = LinkedList(args[i])
        current.set_nelt(nelt)
        current = nelt

    return start


def concat(args):
    '''should take an arbitrary number of lists as arguments and should return a new list representing
    the concatenationoftheselists.'''
    if args[0] == []: return LinkedList('NIL')
    elif len(args) == 1: return args[0].copy()
    current = LinkedList('NIL')

    for i in range(len(args)):
        if args[i].get_elt() != 'NIL': 
            current = args[i].copy()
            for j in range(i, len(args)-1):
                if args[j+1].get_elt() != 'NIL':
                    current.last_elt().set_nelt(args[j+1].copy())
            return current

def car(args):
    '''Take a list as argument and return the first element in that list.'''
    if args[0].get_elt() != 'NIL': return args[0].get_elt() 
    else: raise EvaluationError

def c_map(args):
    '''takes a function and a list as arguments, and it returns a new list containing the results
    of applying the given function to each element of the given list.'''
    f, alist = args[0], args[1]
    if args[0] == 'NIL': f(None)
    start = LinkedList(f([alist.get_elt()]))
    current = start
    for i in range(alist.length()-1): 
        next1 = LinkedList(f([alist.get_nelt().get_elt()]))
        current.set_nelt(next1)
        current, alist = current.get_nelt(), alist.get_nelt()
    return start

def c_filter(args):
    '''takes a function and a list as arguments, and it returns a new list containing only the elements of the
    given list for which the given function returns true'''
    f, alist = args[0], args[1]
    bool_list = c_map(args)

    for i in range(alist.length()):
        if bool_list.elt_at_index(i).get_elt() == '#ص': 
            start = LinkedList(alist.elt_at_index(i).get_elt())
            current = start
            for j in range(i+1, alist.length()):
                if bool_list.elt_at_index(j).get_elt() == '#ص':
                    next1 = LinkedList(alist.elt_at_index(j).get_elt())
                    current.set_nelt(next1)
                    current = current.get_nelt()
            return start
    return LinkedList('NIL')

def c_reduce(args):
    '''takes a function, a list, and an initial value as inputs. It produces its output by successively applying
    the given function to the elements in the list, maintaining an intermediate result along the way.'''
    f, alist, initval = args[0], args[1], args[2]
    while alist:
        initval = f([initval, alist.get_elt()])
        alist = alist.cdr()
    return initval

def evaluate_file(source, env=None):
    '''Read a file and evaluate its content in the given environment.'''
    try:
        envs = Trie('Calree Built-ins')
        envs.keys = carlae_builtins
        if not isinstance(env, Trie):
            if not 'شامل' in envs.children:
                envs.insert_env('شامل')
            env = envs.find_env('شامل')
        text =open(source).read().splitlines()
        for s in text:
            evaluate(parse(tokenize(s)), env)
    except Exception as e: print(e)

def c_let(args):
    '''Create local definitions for variables and evaluate the body in that environment.'''
    try:
        l_vars, body, env = args[0], args[1], args[2]
        child = env.insert_empty_env()
        for var in l_vars:
            value = evaluate(var[1],child)
            evaluate(['عرف']+[var[0]]+[var[1]]+[child], child)
        return evaluate(body, child)
    except Exception as e: print(e)

def c_set(args):
    '''Change the value of an existing variable.'''
    var, value, env = args[0], args[1], args[2]
    _, k_env = env.find_key(var, True)
    return evaluate(['عرف']+[var]+[value]+[k_env], env)

carlae_builtins = {
                    'عرف': define, #define
                    'لامبدا': def_f, #lambda
                    'إذا': lambda args: evaluate(args[1], args[-1]) if evaluate(args[0], args[-1]) == '#ص'\
                                                                   else evaluate(args[2], args[-1]),
                    'و': c_and, #and
                    'أو': c_or, #or
                    'ليس': lambda args: '#ص' if args[0] == '#خ' else '#خ', #not
                    '+': sum,
                    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
                    '*': (lambda a: lambda v: a(a,v))(lambda p,args: 1 if not args else args[0]*p(p, args[1:])),
                    '/': lambda args: args[0]/carlae_builtins['*'](args[1:]),
                    '=?': lambda args: '#ص' if len(set(args)) < 2 else '#خ',
                    '>': def_c(1),
                    '>=': def_c(0, 1),
                    '<': def_c(-1),
                    '<=': def_c(0, -1),
                    'قائمة':build_list, #list
                    'أول':car, #car
                    'آخر':lambda args: args[0].cdr(), #cdr
                    'طول': lambda args: args[0].length(), #length
                    'نسخ': lambda args: args[0].copy(), #copy
                    'سلسل': concat, #concat
                    'عنصر-في-مدخل': lambda args: args[0].elt_at_index(args[1]).get_elt(), #elt-at-index
                    'تعيين': c_map, #map
                    'صف':c_filter, #filter
                    'قلص': c_reduce, #reduce
                    'إبدأ': lambda args: args[-1], #begin
                    'هب': c_let, #let
                    'عد!': c_set, #set!
                     }

carlee_special_forms = set(['عرف', 'لامبدا', 'إذا', 'و', 'أو', 'let', 'set!'])
carlae_laterals = set(['#ص', '#خ', 'NIL'])

def evaluate(tree, env=None):
    """
    Evaluate the given syntax tree according to the rules of the carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # print(tree)
    envs = Trie('Calree Built-ins')
    envs.keys = carlae_builtins
    if not isinstance(env, Trie):
        if not 'شامل' in envs.children: envs.insert_env('شامل')
        env = envs.find_env('شامل')
        if not env: raise EnvironmentError ("البيئة المطلوبة غير موجودة ", env)
    if isinstance(tree, list):
        if not len(tree): return tree
        if len(tree) == 1: 
            if tree[0] == 'قائمة' or 'سلسل': return evaluate(tree[0], env)([[]])
            return evaluate(tree[0], env)
        if len(tree) and isinstance(tree[0], list):
            return evaluate(tree[0], env)([evaluate(e, env) for e in tree[1:]])  
        try: 
            f = env.find_key(tree[0])
            if tree[0] in carlee_special_forms:
                if tree[0] == 'عرف': 
                    if isinstance(tree[1], list):   #Change the arguemnts' representation to suit define's 
                        next1 = ['لامبدا']+[tree[1][1:]]+[tree[2]]+[env]
                        return evaluate([tree[0]]+[tree[1][0]]+[next1], env)
                f = env.root.find_key(tree[0])
                args = tree[1:]
                return f(args+[env])
        except Exception as e:
            pass

        return evaluate(tree[0], env)([evaluate(e, env) for e in tree[1:]])

    try: return env.find_key(tree)
    except KeyError: pass

    if isinstance(tree, float) or isinstance(tree, int) or tree in carlae_laterals: return tree
    raise EvaluationError ('خطأ تقييم للمدخل ', tree)

def result_and_env(tree, env=None):
    envs = Trie('وظائف أولية')
    envs.keys = carlae_builtins
    if not isinstance(env, Trie):
        if not 'شامل' in envs.children:
            envs.insert_env('شامل')
        env = envs.find_env('شامل')
    return (evaluate(tree, env), env)

if __name__ == '__main__':
    print(sys.argv)
    envs = Trie('الوظائف أولية')
    envs.keys = carlae_builtins
    envs.insert_env('شامل')
    env = envs.find_env('شامل')
    for file in sys.argv[1:]: evaluate_file(file, env)
    exit_keys = {"quit", "q", "خ", "خروج"}
    help_keys = {'help', 'مساعده', 'مساعدة'}

    while True:
        source = input('>>> ')
        if source in exit_keys: break
        elif source == "اطبع": env.print_trie()
        elif source == "اطبع -ق": env.print_trie(v=True)
        elif source in help_keys: envs.print_trie(h=True)
        else:
            try:
                print(evaluate(parse(tokenize(source)), env))
            except Exception as e:
                    print('اعتراض مرفوع:', e)
                    print('مدخل غير مقبول. ادخل "مساعده" للمساعدة أو "خروج" للخروج')

