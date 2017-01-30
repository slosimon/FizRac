#######################################################################@@#
# Naloga 1: Kača - Solved for you by slosimon
#######################################################################@@#



##################################################################@005784#
# 1) Sestavite funkcijo kaca(n), kot je zapisano v navodilih.
##################################################################005784@#
import math
def kaca(n):
	a=math.ceil(math.sqrt(n))
	l = []
	for x in range (a):
		ll = []
		if x % 2 == 0:
			for y in range (a):
				#print ((x) * (a)+y+1)
				if (x*a+y+1 <= n):
					ll.append(((x) * (a)+y+1))
				else:
					ll.append(0)
		elif x % 2 == 1:
			for y in range (a):
				#print ((x) * (a+1)-y+3)
				if ( ((x+1) * (a)-y) <= n):
					ll.append(((x+1) * (a)-y))
				else:
					ll.append(0)
		l.append(ll)
	return l









































































































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
                ("kaca(14)",
                 [[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 11, 12],
                  [0, 0, 14, 13]]),
                 ("kaca(13)",
                 [[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 11, 12],
                  [0, 0, 0, 13]]),
                ("kaca(12)",
                 [[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 11, 12],
                  [0, 0, 0, 0]]),
                ("kaca(11)",
                 [[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 11, 0],
                  [0, 0, 0, 0]]),
                ("kaca(10)",
                 [[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 0, 0],
                  [0, 0, 0, 0]]),
                ("kaca(9)",
                 [[1, 2, 3],
                  [6, 5, 4],
                  [7, 8, 9]]),
                ("kaca(8)",
                 [[1, 2, 3],
                  [6, 5, 4],
                  [7, 8, 0]]),
                ("kaca(7)",
                 [[1, 2, 3],
                  [6, 5, 4],
                  [7, 0, 0]]),
                ("kaca(6)",
                 [[1, 2, 3],
                  [6, 5, 4],
                  [0, 0, 0]]),
                ("kaca(5)",
                 [[1, 2, 3],
                  [0, 5, 4],
                  [0, 0, 0]]),
                ("kaca(4)",
                 [[1, 2],
                  [4, 3]]),
                ("kaca(3)",
                 [[1, 2],
                  [0, 3]]),
                ("kaca(2)",
                 [[1, 2],
                  [0, 0]]),
                ("kaca(1)",
                 [[1]]),
                ("kaca(15)",
                 [[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 11, 12],
                  [0, 15, 14, 13]]),
                ("kaca(16)",
                 [[1, 2, 3, 4],
                  [8, 7, 6, 5],
                  [9, 10, 11, 12],
                  [16, 15, 14, 13]]),
                ("kaca(17)",
                 [[1, 2, 3, 4, 5],
                  [10, 9, 8, 7, 6],
                  [11, 12, 13, 14, 15],
                  [0, 0, 0, 17, 16],
                  [0, 0, 0, 0, 0]]),
                ("kaca(101)",
                 [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                  [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12],
                  [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                  [44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34],
                  [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
                  [66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56],
                  [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
                  [88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78],
                  [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 100],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
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
