#######################################################################@@#
# Najdaljše strogo naraščajoče podzaporedje - Solved for you by slosimon
#
#
# Za dano zaporedje celih števil $a = [a_0, a_1, \ldots, a_{n-1}]$ želimo
# najti najdaljše strogo naraščajoče podzaporedje (v nadaljevanju NSNP),
# ne nujno strnjeno.
# 
# Npr. v zaporedju $[0, 8, 4, 12, 2, 10, 6, 14, 1]$ je eno od možnih takih
# NSNP $[0, 4, 10, 14]$. Naj $d(a, i)$ predstavlja dolžino NSNP, ki se
# _konča s členom_ $a_i$. Vemo naslednje:
# $$
# d(a, i) = 1 + \max\{ d(a, j) \mid j < i, a_j < a_i \},
# $$
# pri čemer je maksimum prazne množice enak 0.
#######################################################################@@#

def je_podzaporedje(s, l):
    '''
    Vrni True, če je s podzaporedje zaporedja l, in False sicer.
    '''
    i = 0
    for j in range(len(l)):
        if s[i] == l[j]:
            i += 1
        if i >= len(s):
            return True
    return False

def je_narascajoce(l):
    '''
    Vrni True, če je l strogo naraščajoče zaporedje števil, in False sicer.
    '''
    for i in range(len(l) - 1):
        if l[i] >= l[i + 1]:
            return False
    return True

##################################################################@000517#
# 1) Naivna rekurzivna implementacija funkcije $d(a, i)$ je zelo neučinkovita.
# Sestavite funkcijo dolzina(a), ki vrne tabelo vrednosti
# $[d(a, 0), d(a, 1), \ldots, d(a, n - 1)]$. Funkcija naj bo učinkovita
# tudi za večje tabele. Ocenite časovno zahtevnost svoje funkcije.
# Zgled:
# 
# >>> dolzina([0, 8, 4, 12, 2, 10, 6, 14, 1])
# [1, 2, 2, 3, 2, 3, 3, 4, 2]
##################################################################000517@#
def lis(seq):
    if not seq:
        return seq
    M = [None] * len(seq)
    P = [None] * len(seq)
    L = 1
    M[0] = 0
    for i in range(1, len(seq)):
        lower = 0
        upper = L
        if seq[M[upper-1]] < seq[i]:
            j = upper

        else:
            while upper - lower > 1: #Binary-search - bisekcija
                mid = (upper + lower) // 2
                if seq[M[mid-1]] < seq[i]:
                    lower = mid
                else:
                    upper = mid

            j = lower
        P[i] = M[j-1]
        if j == L or seq[i] < seq[M[j]]:
            M[j] = i
            L = max(L, j+1)
    result = []
    pos = M[L-1]
    for _ in range(L):
        result.append(seq[pos])
        pos = P[pos]

    return result[::-1]

def length(l,a):
    lista=l[0:a+1]
    li=[]
    b=lista[a]
    for x in lista:
        if (x <= b):
            li.append(x)
    return (lis(li))

def dolzina(l):
    g=[]
    for i in range (len(l)):
        g.append(len(length(l,i)))
    return g

##################################################################@000518#
# 2) Sestavite funkcijo nsnp(a), ki vrne enega od NSNP-jev za tabelo a.
# 
# Funkcija naj najprej izračuna tabeli
# $[d(a, 0), d(a, 1), \ldots, d(a, n-1)]$ in
# $j = [j_0, j_1, \ldots, j_{n-1}]$,
# pri čemer je $j_k$ tisti indeks, pri katerem je dosežen maksimum v
# definiciji $d(a, k)$. Poseben primer je $j_k = \texttt{None}$, če je
# maksimum v definiciji prazen. Zaporedje
# $$
# a[i], a[j[i]], a[j[j[i]]], a[j[j[j[i]]]], \ldots
# $$
# ki se ustavi, ko naletimo na $j[j[\cdots]] = \texttt{None}$, je najdaljše
# naraščajoče podzaporedje, ki se konča z $a[i]$, zapisano v _obratnem
# vrstnem redu_. Njegova dolžina je $d(a, i)$. S pomočjo tega lahko hitro
# poiščemo NSNP za tabelo $a$.
# 
# Zgled:
# 
# >>> nsnp([0, 8, 4, 12, 2, 10, 6, 14, 1])
# [0, 4, 10, 14]
##################################################################000518@#

def nsnp(a): #Just use longest increasing subsequence (lis)
    return lis(a)









































































































#######################################################################@@#
# Kode pod to črto nikakor ne spreminjajte.
##########################################################################

"TA VRSTICA JE PRAVILNA."
"ČE VAM PYTHON SPOROČI, DA JE V NJEJ NAPAKA, SE MOTI."
"NAPAKA JE NAJVERJETNEJE V ZADNJI VRSTICI VAŠE KODE."
"ČE JE NE NAJDETE, VPRAŠAJTE ASISTENTA."



























































import io, json, os, re, sys, shutil, traceback
from contextlib import contextmanager
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
class Check:
    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part['errors'] = []
            part['challenge'] = []
        Check.current = None
        Check.part_counter = None
        Check.user_id = None

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current = Check.parts[Check.part_counter]
        return Check.current.get('solution', '').strip() != ''

    @staticmethod
    def error(msg, *args, **kwargs):
        Check.current['errors'].append(msg.format(*args, **kwargs))

    @staticmethod
    def challenge(x, k=""):
        pair = (str(k), str(Check.canonize(x)))
        Check.current['challenge'].append(pair)

    @staticmethod
    def execute(example, env={}, use_globals=True, do_eval=False, catch_exception=False):
        local_env = {}
        local_env.update(env)
        global_env = globals() if use_globals else {}
        old_stdout, old_stderr = sys.stdout, sys.stderr
        new_stdout, new_stderr = io.StringIO(), io.StringIO()
        exec_info = {'env': local_env}
        try:
            sys.stdout, sys.stderr = new_stdout, new_stderr
            if do_eval:
                exec_info['value'] = eval(example, global_env, local_env)
            else:
                exec(example, global_env, local_env)
        except Exception as e:
            if catch_exception:
                exec_info['exception'] = e
            else:
                raise e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        exec_info['stdout'] = new_stdout.getvalue()
        exec_info['stderr'] = new_stderr.getvalue()
        return exec_info

    @staticmethod
    def run(example, state, message=None, env={}, clean=lambda x: x):
        code = "\n".join(example)
        example = "  >>> " + "\n  >>> ".join(example)
        s = {}
        s.update(env)
        exec (code, globals(), s)
        errors = []
        for (x,v) in state.items():
            if x not in s:
                errors.append('morajo nastaviti spremenljivko {0}, vendar je ne'.format(x))
            elif clean(s[x]) != clean(v):
                errors.append('morajo nastaviti {0} na {1},\nvendar nastavijo {0} na {2}'.format(x, v, s[x]))
        if errors:
            Check.error('Ukazi\n{0}\n{1}.', example,  ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    def canonize(x, digits=6):
        if   type(x) is float:
            x = round(x, digits)
            # We want to canonize -0.0 and similar small negative numbers to 0.0
            # Since -0.0 still behaves as False, we can use the following
            return x if x else 0.0
        elif type(x) is complex: return complex(Check.canonize(x.real, digits), Check.canonize(x.imag, digits))
        elif type(x) is list: return list([Check.canonize(y, digits) for y in x])
        elif type(x) is tuple: return tuple([Check.canonize(y, digits) for y in x])
        elif type(x) is dict: return sorted([(Check.canonize(k, digits), Check.canonize(v, digits)) for (k,v) in x.items()])
        elif type(x) is set: return sorted([Check.canonize(y, digits) for y in x])
        else: return x

    @staticmethod
    def equal(example, value=None, exception=None,
                clean=lambda x: x, env={},
                precision=1.0e-6, strict_float=False, strict_list=True):
        def difference(x, y):
            if x == y: return None
            elif (type(x) != type(y) and
                 (strict_float or not (type(y) in [int, float, complex] and type(x) in [int, float, complex])) and
                 (strict_list or not (type(y) in [list, tuple] and type(x) in [list, tuple]))):
                return "različna tipa"
            elif type(y) in [int, float, complex]:
                return ("numerična napaka" if abs(x - y) > precision else None)
            elif type(y) in [tuple,list]:
                if len(y) != len(x): return "napačna dolžina seznama"
                else:
                    for (u, v) in zip(x, y):
                        msg = difference(u, v)
                        if msg: return msg
                    return None
            elif type(y) is dict:
                if len(y) != len(x): return "napačna dolžina slovarja"
                else:
                    for (k, v) in y.items():
                        if k not in x: return "manjkajoči ključ v slovarju"
                        msg = difference(x[k], v)
                        if msg: return msg
                    return None
            else: return "različni vrednosti"

        local = locals()
        local.update(env)

        if exception:
            try:
                returned = eval(example, globals(), local)
            except Exception as e:
                if e.__class__ != exception.__class__ or e.args != exception.args:
                    Check.error("Izraz {0} sproži izjemo {1!r} namesto {2!r}.",
                                example, e, exception)
                    return False
                else:
                    return True
            else:
                Check.error("Izraz {0} vrne {1!r} namesto da bi sprožil izjemo {2}.",
                            example, returned, exception)
                return False

        else:
            returned = eval(example, globals(), local)
            reason = difference(clean(returned), clean(value))
            if reason:
                Check.error("Izraz {0} vrne {1!r} namesto {2!r} ({3}).",
                            example, returned, value, reason)
                return False
            else:
                return True

    @staticmethod
    def generator(example, good_values, should_stop=False, further_iter=0, env={},
                  clean=lambda x: x, precision=1.0e-6, strict_float=False, strict_list=True):
        from types import GeneratorType
        
        def difference(x, y):
            if x == y: return None
            elif (type(x) != type(y) and
                    (strict_float or not (type(y) in [int, float, complex] and type(x) in [int, float, complex])) and
                    (strict_list or not (type(y) in [list, tuple] and type(x) in [list, tuple]))):
                return "različna tipa"
            elif type(y) in [int, float, complex]:
                return ("numerična napaka" if abs(x - y) > precision else None)
            elif type(y) in [tuple,list]:
                if len(y) != len(x): return "napačna dolžina seznama"
                else:
                    for (u, v) in zip(x, y):
                        msg = difference(u, v)
                        if msg: return msg
                    return None
            elif type(y) is dict:
                if len(y) != len(x): return "napačna dolžina slovarja"
                else:
                    for (k, v) in y.items():
                        if k not in x: return "manjkajoči ključ v slovarju"
                        msg = difference(x[k], v)
                        if msg: return msg
                    return None
            else: return "različni vrednosti"

        local = locals()
        local.update(env)
        gen = eval(example, globals(), local)
        if not isinstance(gen, GeneratorType):
            Check.error("{0} ni generator.", example)
            return False

        iter_counter = 0
        try:
            for correct_ans in good_values:
                iter_counter += 1
                student_ans = gen.__next__()
                reason = difference(clean(correct_ans), clean(student_ans))
                if reason:
                    Check.error("Element #{0}, ki ga vrne generator {1} ni pravilen: {2!r} namesto {3!r} ({4}).",
                                iter_counter, example, student_ans, correct_ans, reason)
                    return False
            for i in range(further_iter):
                iter_counter += 1
                gen.__next__() # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", example)
            return False
        
        if should_stop:
            try:
                gen.__next__()
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", example)
            except StopIteration:
                pass # this is fine
        return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        with open(filename, "w", encoding=encoding) as _f:
            for line in content:
                print(line, file=_f)
        old_errors = Check.current["errors"][:]
        yield
        new_errors = Check.current["errors"][len(old_errors):]
        Check.current["errors"] = old_errors
        if new_errors:
            new_errors = ["\n    ".join(error.split("\n")) for error in new_errors]
            Check.error("Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}", filename, "\n  ".join(content), "\n- ".join(new_errors))

    @staticmethod
    def out_file(filename, content, encoding=None):
        with open(filename, encoding=encoding) as _f:
            out_lines = _f.readlines()
        len_out, len_given = len(out_lines), len(content)
        if len_out < len_given:
            out_lines += (len_given - len_out) * ["\n"]
        else:
            content += (len_out - len_given) * ["\n"]
        equal = True
        line_width = max(len(out_line.rstrip()) for out_line in out_lines + ["je enaka"])
        diff = []
        for out, given in zip(out_lines, content):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append("{0} {1} {2}".format(out.ljust(line_width), "|" if out == given else "*", given))
        if not equal:
            Check.error("Izhodna datoteka {0}\n je enaka{1}  namesto:\n  {2}", filename, (line_width - 7) * " ", "\n  ".join(diff))
            return False
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not part['solution'].strip():
                print('Podnaloga {0} je brez rešitve.'.format(i + 1))
            elif part['errors']:
                print('Podnaloga {0} ni prestala vseh testov:'.format(i + 1))
                for e in part['errors']:
                    print("- {0}".format("\n  ".join(e.splitlines())))
            elif 'rejection' in part:
                reason = ' ({0})'.format(part['rejection']) if part['rejection'] else ''
                print('Podnaloga {0} je zavrnjena.{1}'.format(i + 1, reason))
            else:
                print('Podnaloga {0} je pravilno rešena.'.format(i + 1))

def _check():
    _filename = os.path.abspath(sys.argv[0])
    with open(_filename, encoding='utf-8') as _f:
        _source = _f.read()

    Check.initialize([
        {
            'part': int(match.group('part')),
            'solution': match.group('solution')
        } for match in re.compile(
            r'#+@(?P<part>\d+)#\n' # beginning of header
            r'.*?'                 # description
            r'#+(?P=part)@#\n'     # end of header
            r'(?P<solution>.*?)'   # solution
            r'(?=\n#+@)',          # beginning of next part
            flags=re.DOTALL|re.MULTILINE
        ).finditer(_source)
    ])
    Check.parts[-1]['solution'] = Check.parts[-1]['solution'].rstrip()


    problem_match = re.search(
        r'#+@@#\n'           # beginning of header
        r'.*?'               # description
        r'#+@@#\n'           # end of header
        r'(?P<preamble>.*?)' # preamble
        r'(?=\n#+@)',        # beginning of first part
        _source, flags=re.DOTALL|re.MULTILINE)

    if not problem_match:
        print("NAPAKA: datoteka ni pravilno oblikovana")
        sys.exit(1)

    _preamble = problem_match.group('preamble')

    
    if Check.part():
        try:
            test_data = [
                ("""dolzina([0, 8, 4, 12, 2, 10, 6, 14, 1])""", [1, 2, 2, 3, 2, 3, 3, 4, 2]),
                ("""dolzina([0, 8, 4, 12, 2, 10, 6])""", [1, 2, 2, 3, 2, 3, 3]),
                ("""dolzina([33, 36, 12, 40, 19, 34, 16, 1, 7, 1, 6, 5, 9, 32, 19, 1, 35, 35, 7, 16])""",
                 [1, 2, 1, 3, 2, 3, 2, 1, 2, 1, 2, 2, 3, 4, 4, 1, 5, 5, 3, 4]),
                ("""dolzina([24, 8, 44, 43, 32, 45, 15, 16, 15, 38, 16, 5, 18, 42, 54, 2, 44, 24, 1, 57, 9, 37, 9, 37, 26, 16, 53, 40, 47, 29, 13, 15, 21, 44, 40, 52, 21, 34, 53, 1])""",
                 [1, 1, 2, 2, 2, 3, 2, 3, 2, 4, 3, 1, 4, 5, 6, 1, 6, 5, 1, 7, 2, 6, 2, 6, 6, 3, 7, 7, 8, 7, 3, 4, 5, 8, 8, 9, 5, 8, 10, 1]),
                ("""dolzina([38, 11, 2, 14, 39, 15, 28, 1, 35, 6, 30, 26, 17, 32, 31, 37, 5, 29, 24, 25, 10, 20, 23, 3, 34, 7, 4, 18, 8, 21, 13, 19, 33, 9, 22, 40, 36, 12, 16, 27])""",
                 [1, 1, 1, 2, 3, 3, 4, 1, 5, 2, 5, 4, 4, 6, 6, 7, 2, 5, 5, 6, 3, 5, 6, 2, 7, 3, 3, 5, 4, 6, 5, 6, 7, 5, 7, 8, 8, 6, 7, 8]),
                ("""dolzina([4])""", [1]),
                ("""dolzina([])""", []),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            pass
        except:
            Check.error("Testi sprožijo izjemo\n  {0}", "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    
    if Check.part():
        try:
            test_data = [
                (["a = [0, 8, 4, 12, 2, 10, 6, 14, 1]",
                  "b = nsnp(a[:])",
                  "dolzina = len(b)",
                  "ok = je_podzaporedje(b, a) and je_narascajoce(b)"],
                 {'dolzina': 4, 'ok': True}), # e.g. [0, 8, 12, 14]
                (["a = [0, 8, 4, 12, 2, 10, 6]",
                  "b = nsnp(a[:])",
                  "dolzina = len(b)",
                  "ok = je_podzaporedje(b, a) and je_narascajoce(b)"],
                 {'dolzina': 3, 'ok': True}), # e.g. [0, 8, 12]
                (["a = [33, 36, 12, 40, 19, 34, 16, 1, 7, 1, 6, 5, 9, 32, 19, 1, 35, 35, 7, 16]",
                  "b = nsnp(a[:])",
                  "dolzina = len(b)",
                  "ok = je_podzaporedje(b, a) and je_narascajoce(b)"],
                 {'dolzina': 5, 'ok': True}), # e.g. [1, 7, 9, 32, 35]
                (["a = [24, 8, 44, 43, 32, 45, 15, 16, 15, 38, 16, 5, 18, 42, 54, 2, 44, 24, 1, 57, 9, 37, 9, 37, 26, 16, 53, 40, 47, 29, 13, 15, 21, 44, 40, 52, 21, 34, 53, 1]",
                  "b = nsnp(a[:])",
                  "dolzina = len(b)",
                  "ok = je_podzaporedje(b, a) and je_narascajoce(b)"],
                 {'dolzina': 10, 'ok': True}), # e.g. [8, 15, 16, 18, 24, 37, 40, 47, 52, 53]
                (["a = [38, 11, 2, 14, 39, 15, 28, 1, 35, 6, 30, 26, 17, 32, 31, 37, 5, 29, 24, 25, 10, 20, 23, 3, 34, 7, 4, 18, 8, 21, 13, 19, 33, 9, 22, 40, 36, 12, 16, 27]",
                  "b = nsnp(a[:])",
                  "dolzina = len(b)",
                  "ok = je_podzaporedje(b, a) and je_narascajoce(b)"],
                 {'dolzina': 8, 'ok': True}), # e.g. [11, 14, 15, 28, 30, 32, 37, 40]
                (["a = [4]",
                  "b = nsnp(a)"],
                 {'b': [4]}),
                (["a = []",
                  "b = nsnp(a)"],
                 {'b': []}),
            ]
            for td in test_data:
                if not Check.run(*td):
                    break
            pass
        except:
            Check.error("Testi sprožijo izjemo\n  {0}", "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    

    
    Check.summarize()
    print('Naloge rešujete kot anonimni uporabnik, zato rešitve niso shranjene.')
    

_check()

#####################################################################@@#
