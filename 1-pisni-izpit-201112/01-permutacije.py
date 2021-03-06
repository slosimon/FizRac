#######################################################################@@#
# Permutacije - Solved for you by slosimon
#
# Permutacije reda $n$ so bijektivne preslikave na množici
# $\{0, \ldots, n - 1\}$, ki jih podamo s seznamom slik.
# 
# Npr. [2, 0, 1] predstavlja permutacijo, ki slika $0 \mapsto 2$,
# $1 \mapsto 0$ in $2 \mapsto 1$; seznam [3, 0, 2, 4, 1] pa predstavlja
# permutacijo, ki slika $0 \mapsto 3$, $1 \mapsto 0$, $2 \mapsto 2$,
# $3 \mapsto 4$ in $4 \mapsto 1$.
# 
# Razredu Permutacija bomo dodali konstruktor in še dve metodi. Seveda
# lahko definirate še kakšne druge metode, ki olajšajo delo, na primer
# metodo __repr__(self).
# 
# _Opomba:_ Obstoječemu razredu lahko dodate novo metodo takole (razred
# FooBar že obstaja):
# 
# class FooBar(FooBar):
# 
#     def nova_metoda(self, baz):
#         pass
#######################################################################@@#

class Permutacija:
    
	pass

##################################################################@000512#
# 1) Sestavite konstruktor __init__(self, s), ki ustvari permutacijo iz
# podanega seznama slik. Konstruktor naj v atribut s shrani kopijo
# seznama, ki je podan kot argument. Zgled:
# 
# >>> l = [3, 0, 2, 4, 1]
# >>> p = Permutacija(l)
# >>> l.append(5)
# >>> p.s
# [3, 0, 2, 4, 1]
# 
# _Pozor:_ Če se po klicu konstruktorja seznam, ki je bil podan kot
# argument, spremeni, se s tem objekt ne sme spremeniti.
##################################################################000512@#
	
	def __init__(self,s):
		l=[]
		for x in s: #lahko nadomestimo s hardcopy
			l.append(x)
		self.s=l

##################################################################@000513#
# 2) Sestavite metodo __call__(self, m), ki vrne sliko števila $m$ (če je
# $m$ prevelik ali premajhen, naj vrže izjemo). Zgled:
# 
# >>> p = Permutacija([3, 0, 2, 4, 1])
# >>> p(3)
# 4
##################################################################000513@#

	def __call__(self,m):
			return self.s[m]

##################################################################@000514#
# 3) Sestavite metodo inverz(self), ki vrne inverzno permutacijo. Zgled:
# 
# >>> p = Permutacija([3, 0, 2, 4, 1])
# >>> p_inv = p.inverz()
# >>> p_inv.s
# [1, 4, 2, 0, 3]
##################################################################000514@#

	def inverz(self):
		l=[]
		li=self.s
		for x in range (0,len(li)): #če kdo pozna dark magic za inverz, nej pove
			for y in range (0, len(li)):
				if (li[y]==x):
					l.append(y)
		p=Permutacija(l)
		return p









































































































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
                ('Permutacija([3, 0, 2, 4, 1]).s', [3, 0, 2, 4, 1]),
                ('Permutacija([0]).s', [0]),
                ('Permutacija([0, 1, 2]).s', [0, 1, 2]),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8]).s', [4, 0, 6, 3, 1, 5, 2, 7, 8]),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]).s', [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20]).s', [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20]),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14]).s', [0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14]),
            ]
            additional_tests = [
                (["l = [3, 0, 2, 4, 1]",
                  "p = Permutacija(l)",
                  "l.extend([13, 14, 15])",
                  "k = p.s"],
                 {'k': [3, 0, 2, 4, 1]}),
                (["l = [4, 0, 6, 3, 1, 5, 2, 7, 8]",
                  "p = Permutacija(l)",
                  "del l[0]",
                  "del l[-1]", 
                  "k = p.s"],
                 {'k': [4, 0, 6, 3, 1, 5, 2, 7, 8]}),
            ]
            vse_ok = True
            for td in test_data:
                if not Check.equal(*td):
                    vse_ok = False
                    break
            if vse_ok:
                for td in additional_tests:
                    if Check.run(*td):
                        break
            pass
        except:
            Check.error("Testi sprožijo izjemo\n  {0}", "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    
    if Check.part():
        try:
            test_data = [
                ('Permutacija([3, 0, 2, 4, 1])(0)', 3),
                ('Permutacija([3, 0, 2, 4, 1])(1)', 0),
                ('Permutacija([3, 0, 2, 4, 1])(2)', 2),
                ('Permutacija([3, 0, 2, 4, 1])(3)', 4),
                ('Permutacija([3, 0, 2, 4, 1])(4)', 1),
                ('Permutacija([0])(0)', 0),
                ('Permutacija([0, 1, 2])(0)', 0),
                ('Permutacija([0, 1, 2])(1)', 1),
                ('Permutacija([0, 1, 2])(2)', 2),
                ('Permutacija([2, 1, 0])(0)', 2),
                ('Permutacija([2, 1, 0])(1)', 1),
                ('Permutacija([2, 1, 0])(2)', 0),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(0)', 4),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(1)', 0),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(2)', 6),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(3)', 3),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(4)', 1),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(5)', 5),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(6)', 2),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(7)', 7),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8])(8)', 8),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2])(0)', 4),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2])(1)', 0),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2])(2)', 6),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2])(3)', 3),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2])(4)', 1),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2])(5)', 5),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2])(6)', 2),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(0)', 1),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(1)', 0),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(2)', 3),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(3)', 2),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(4)', 5),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(5)', 4),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(6)', 7),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(7)', 6),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(8)', 9),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(9)', 8),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(10)', 11),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(11)', 10),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(12)', 13),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12])(13)', 12),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(0)', 1),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(1)', 0),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(2)', 3),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(3)', 2),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(4)', 5),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(5)', 4),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(6)', 7),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(7)', 6),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(8)', 9),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(9)', 8),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(10)', 11),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(11)', 10),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(12)', 13),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(13)', 12),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(14)', 14),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(15)', 15),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(16)', 16),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(17)', 17),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(18)', 18),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(19)', 19),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20])(20)', 20),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(0)', 0),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(1)', 13),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(2)', 6),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(3)', 5),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(4)', 2),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(5)', 10),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(6)', 8),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(7)', 1),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(8)', 12),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(9)', 9),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(10)', 11),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(11)', 3),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(12)', 4),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(13)', 7),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(14)', 17),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(15)', 15),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(16)', 16),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14])(17)', 14),
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
                ('Permutacija([3, 0, 2, 4, 1]).inverz().s', [1, 4, 2, 0, 3]),
                ('Permutacija([0]).inverz().s', [0]),
                ('Permutacija([0, 1, 2]).inverz().s', [0, 1, 2]),
                ('Permutacija([2, 1, 0]).inverz().s', [2, 1, 0]),
                ('Permutacija([4, 0, 6, 3, 1, 5, 2, 7, 8]).inverz().s', [1, 4, 6, 3, 0, 5, 2, 7, 8]),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]).inverz().s', [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]),
                ('Permutacija([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20]).inverz().s', [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14, 15, 16, 17, 18, 19, 20]),
                ('Permutacija([0, 13, 6, 5, 2, 10, 8, 1, 12, 9, 11, 3, 4, 7, 17, 15, 16, 14]).inverz().s', [0, 7, 4, 11, 12, 3, 2, 13, 6, 9, 5, 10, 8, 1, 17, 15, 16, 14]),
            ]
            vse_ok = True
            for td in test_data:
                if not Check.equal(*td):
                    vse_ok = False
                    break
            if vse_ok:
                Check.equal('type(Permutacija([3, 0, 2, 4, 1]).inverz()) is Permutacija', True)
            pass
        except:
            Check.error("Testi sprožijo izjemo\n  {0}", "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    

    
    Check.summarize()
    print('Naloge rešujete kot anonimni uporabnik, zato rešitve niso shranjene.')
    

_check()

#####################################################################@@#
