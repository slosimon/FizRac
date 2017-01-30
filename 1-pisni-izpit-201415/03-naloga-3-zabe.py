#######################################################################@@#
# Naloga 3: Žabe - Solved for you by slosimon
#######################################################################@@#



##################################################################@005295#
# 1) Sestavite funkcijo pojedina(m), kot je zapisano v navodilih.
##################################################################005295@#
import queue

def pojedina(m):
	l = [0] * len(m)
	o = queue.PriorityQueue()
	l[0] = m[0]
	o.put(0, 0)
	maximum = 0
	while not o.empty():
		a = o.get()
		if a+2 < len(m):
			if l[a+2] < l[a] + m[a+2]:
				l[a+2] = l[a]+m[a+2]
				o.put(a+2, a+2)
		if a+3 < len(m):
			if l[a+3] < l[a] + m[a+3]:
				l[a+3] = l[a]+m[a+3]
				o.put(a+3, a+3)
	return max(l)
		

##################################################################@005296#
# 2) Če je časovna zahtevnost funkcije iz prejšnje podnaloge že $O(n)$, vam
# te naloge ni treba reševati. Če že imate delujočo počasno rešitev, lahko
# na tem mestu poskušate rešiti nalogo še na učinkovit način.
# 
# Če območja za rešitev ne pustite praznega (npr. vnesete kak komentar),
# bo Tomo vašo rešitev iz prejšnje podnaloge preveril še na večjih testnih
# primerih. Če se testiranje ne izvede v doglednem času, časovna zahtevnost
# vaše rešitve ni $O(n)$.
##################################################################005296@#

#rešeno v O(n) že v prvo

def pojedina(m):
	l = [0] * len(m)
	o = queue.PriorityQueue()
	l[0] = m[0]
	o.put(0, 0)
	maximum = 0
	while not o.empty():
		a = o.get()
		if a+2 < len(m):
			if l[a+2] < l[a] + m[a+2]:
				l[a+2] = l[a]+m[a+2]
				o.put(a+2, a+2)
		if a+3 < len(m):
			if l[a+3] < l[a] + m[a+3]:
				l[a+3] = l[a]+m[a+3]
				o.put(a+3, a+3)
	return max(l)







































































































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
                ("""pojedina([1, 2, 0, 3, 4, 5, 1])""", 9),
                ("""pojedina([1, 2, 3, 4, 5, 6, 7, 8, 9])""", 25),
                ("""pojedina([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])""", 0),
                ("""pojedina([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])""", 5),
                ("""pojedina([1, 1, 3, 1, 1, 4, 1, 3, 1, 1])""", 12),
                ("""pojedina([1, 1, 3, 1, 1, 4, 1, 3, 1])""", 11),
                ("""pojedina([1, 1, 4, 1, 5, 1, 4, 1, 1, 7])""", 21),
                ("""pojedina([1, 1, 1, 5, 1, 1, 5, 1, 1, 5])""", 16),
                ("""pojedina([1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 37])""", 49),
                ("""pojedina([1, 8, 14, 10, 11, 13, 8, 13, 11, 9])""", 50),
                ("""pojedina([16, 13, 19, 5, 2, 6, 17, 20, 11, 0])""", 65),
                ("""pojedina([3, 8, 14, 1, 2, 13, 13, 11, 6, 0])""", 41),
                ("""pojedina([7, 19, 9, 13, 2, 12, 12, 4, 8, 11])""", 47),
                ("""pojedina([3, 8, 7, 12, 19, 6, 5, 2, 0, 2])""", 36),
                ("""pojedina([6, 13, 0, 6, 6, 0, 3, 3, 16, 2])""", 31),
                ("""pojedina([1, 13, 5, 6, 7, 13, 2, 10, 4, 17])""", 47),
                ("""pojedina([13, 6, 5, 1, 7, 3, 17, 15, 7, 18])""", 60),
                ("""pojedina([4, 10, 7, 17, 15, 14, 5, 4, 7, 10])""", 49),
                ("""pojedina([1, 2])""", 1),
                ("""pojedina([0, 2, 3])""", 3),
                ("""pojedina([1])""", 1),
                ("""pojedina([0])""", 0),
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
                ("""pojedina([1, 2, 0, 3, 4, 5, 1])""", 9),
                ("""pojedina([1, 2, 3, 4, 5, 6, 7, 8, 9])""", 25),
                ("""pojedina([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])""", 0),
                ("""pojedina([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])""", 5),
                ("""pojedina([1, 1, 3, 1, 1, 4, 1, 3, 1, 1])""", 12),
                ("""pojedina([1, 8, 14, 10, 11, 13, 8, 13, 11, 9])""", 50),
                ("""pojedina([16, 13, 19, 5, 2, 6, 17, 20, 11, 0])""", 65),
                ("""pojedina([1, 2])""", 1),
                ("""pojedina([0, 2, 3])""", 3),
                ("""pojedina([28, 28, 17, 10, 1, 5, 10, 30, 1, 2, 13, 27, 12, 20, 0, 20, 24, 2, 18, 21, 30, 16, 4, 2, 2, 8, 5, 21, 11, 6, 20, 23, 14, 30, 13, 4, 28, 2, 22, 15, 5, 23, 22, 8, 26, 0, 0, 9, 27, 22, 7, 29, 30, 14, 9, 11, 8, 27, 4, 0, 19, 14, 0, 19, 13, 1, 5, 16, 24, 4, 18, 5, 14, 20, 11, 23, 4, 16, 14, 9, 25, 16, 3, 28, 9, 25, 13, 16, 12, 1, 10, 23, 27, 2, 23, 30, 28, 28, 24, 12])""", 858),
                ("""pojedina([7, 6, 1, 10, 8, 5, 10, 3, 3, 4, 6, 0, 5, 7, 3, 6, 0, 2, 2, 5, 10, 2, 10, 3, 4, 10, 5, 1, 1, 10, 8, 10, 1, 2, 7, 10, 6, 10, 1, 4, 8, 4, 2, 7, 5, 9, 7, 1, 9, 3, 7, 8, 4, 2, 2, 0, 10, 1, 8, 3, 3, 10, 5, 3, 3, 9, 2, 10, 9, 2, 4, 3, 10, 0, 5, 3, 9, 10, 10, 6, 0, 9, 6, 7, 1, 3, 6, 8, 7, 3, 10, 4, 0, 0, 10, 1, 7, 6, 0, 6])""", 320),
                ("""pojedina([3, 3, 3, 6, 1, 0, 0, 4, 10, 7, 10, 0, 4, 2, 5, 0, 1, 0, 8, 6, 9, 9, 4, 2, 2, 8, 4, 8, 3, 5, 10, 8, 0, 8, 8, 4, 4, 1, 7, 4, 1, 1, 2, 4, 2, 10, 3, 9, 7, 3, 3, 2, 9, 0, 3, 7, 7, 2, 1, 9, 5, 10, 7, 7, 4, 10, 4, 0, 5, 1, 7, 0, 7, 8, 0, 3, 5, 5, 0, 1, 8, 5, 0, 7, 9, 10, 9, 2, 0, 1, 4, 4, 7, 5, 3, 5, 6, 10, 0, 3, 8, 9, 0, 9, 10, 9, 1, 0, 1, 6, 2, 8, 6, 1, 4, 8, 8, 7, 2, 10, 3, 2, 9, 8, 4, 4, 4, 9, 10, 2, 7, 2, 3, 1, 9, 6, 7, 10, 5, 0, 5, 1, 0, 2, 5, 9, 2, 7, 2, 4])""", 426),
                ("""pojedina([9, 16, 5, 9, 15, 10, 11, 10, 15, 8, 18, 10, 8, 8, 6, 18, 6, 19, 17, 9, 7, 19, 9, 6, 19, 17, 10, 20, 12, 8, 14, 9, 11, 14, 15, 5, 9, 6, 20, 16, 13, 18, 17, 20, 20, 12, 15, 6, 6, 10, 20, 7, 16, 15, 14, 8, 20, 12, 6, 16, 15, 18, 10, 15, 17, 20, 11, 10, 11, 19, 5, 7, 11, 15, 13, 18, 18, 14, 17, 13, 15, 16, 9, 13, 10, 17, 13, 10, 5, 12, 17, 8, 6, 19, 10, 19, 6, 11, 7, 7, 12, 15, 16, 15, 7, 8, 12, 6, 17, 7, 15, 15, 13, 12, 15, 13, 12, 17, 12, 10, 6, 6, 12, 11, 8, 12, 16, 8, 10, 9, 8, 13, 13, 18, 15, 17, 11, 11, 5, 19, 18, 5, 9, 6, 7, 19, 17, 12, 6, 12, 11, 13, 19, 20, 5, 15, 15, 20, 6, 14, 17, 5, 20, 20, 18, 20, 5, 15, 17, 11, 11, 10, 13, 5, 18, 5, 5, 17, 14, 9, 5, 18, 10, 16, 16, 18, 7, 15, 8, 7, 14, 18, 12, 11, 13, 5, 14, 6, 16, 18])""", 1357),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            pass
        except:
            Check.error("Testi sprožijo izjemo\n  {0}", "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    

    
    Check.summarize()
    print('Naloge rešujete kot anonimni uporabnik, zato rešitve niso shranjene.')
    

_check()

#####################################################################@@#
