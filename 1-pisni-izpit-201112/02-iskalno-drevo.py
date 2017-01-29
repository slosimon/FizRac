#######################################################################@@#
# Iskalno drevo 
#
# Iskalno drevo je podano z razredom IskalnoDrevo. Konstruktor je že
# implementiran. Kot argument dobi konstruktor seznam števil, ki jih enega
# za drugim vstavi v iskalno drevo, tako da kliče metodo dodaj. Poleg
# tega ima razred IskalnoDrevo še dve metodi in sicer:
# 
# * metodo pravilno, ki preveri, če je to drevo res pravilno iskalno
#   drevo, in
# * metodo drevo, ki vrne to drevo kot objekt razreda Drevo (navadno
#   dvojiško drevo).
# 
# Zgled:
# 
# >>> d = IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6])
# >>> d.drevo()
# Drevo(3,
#       levo=Drevo(2,
#                  levo=Drevo(1)),
#       desno=Drevo(9,
#                   levo=Drevo(4,
#                              desno=Drevo(8,
#                                          levo=Drevo(7,
#                                                     levo=Drevo(6))))))
#######################################################################@@#

class Drevo:

    def __init__(self, *args, **kwargs):
        if args:
            self.prazno = False
            self.vsebina = args[0]
            self.levo = kwargs.get('levo', Drevo())
            self.desno = kwargs.get('desno', Drevo())
        else:
            self.prazno = True

    def __repr__(self, zamik=''):
        if self.prazno:
            return 'Drevo()'.format(zamik)
        else:
            ret = 'Drevo({0}'.format(self.vsebina) + \
                (',\n{0}      levo={1}'.format(
                    zamik,
                    self.levo.__repr__(zamik + '           ')
                ) if not self.levo.prazno else '') + \
                (',\n{0}      desno={1}'.format(
                    zamik,
                    self.desno.__repr__(zamik + '            ')
                ) if not self.desno.prazno else '') + \
                ')'    
            return ret


class IskalnoDrevo:

    def __init__(self, vsebina=[]):
        self.prazno = True
        for n in vsebina:
            self.dodaj(n)

    def drevo(self, zamik=''):
        if self.prazno:
          return Drevo()
        else:
          return Drevo(self.vsebina,
                       levo=self.levo.drevo(),
                       desno=self.desno.drevo())

    def pravilno(self, minimum=None, maksimum=None):
        if self.prazno:
            return True
        elif minimum and self.vsebina < minimum:
            return False
        elif maksimum and self.vsebina > maksimum:
            return False
        else:
            return (self.levo.pravilno(minimum, self.vsebina) and
                    self.desno.pravilno(self.vsebina, maksimum))

    def dodaj(self, podatek):
        if self.prazno:
            self.prazno = False
            self.vsebina = podatek
            self.levo = IskalnoDrevo()
            self.desno = IskalnoDrevo()
        elif self.vsebina > podatek:
            self.levo.dodaj(podatek)
        elif self.vsebina < podatek:
            self.desno.dodaj(podatek)

##################################################################@000515#
# 1) Razredu IskalnoDrevo dodajte metodo prestej_manjse(self, n), ki
# prešteje, koliko vozlišč v iskalnem drevesu ima vsebino manjšo od n.
# Zgled:
# 
# >>> d = IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6])
# >>> d.prestej_manjse(5)
# 4
# 
# _Pozor:_ Ni nujno, da se n pojavi v drevesu!
##################################################################000515@#

    def prestej_manjse(self,n):
        d=self
        if (not d.prazno):
            a=0
            b=0
            try:
                a=IskalnoDrevo.prestej_manjse(d.levo,n)
            except AttributeError:
                pass   
            try:
                b=IskalnoDrevo.prestej_manjse(d.desno,n)
            except AttributeError: 
                pass  
            if (d.vsebina < n):
                return (1+a+b)
            else:
                return (a+b)
            
        else:
            return 0

##################################################################@000516#
# 2) Razredu IskalnoDrevo dodajte metodo vsota_interval(self, a, b), ki
# izračuna vsoto vseh elementov $x$ v drevesu self, za katere velja
# $a \leq x \leq b$. Zgled:
# 
# >>> d = IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6])
# >>> d.vsota_interval(2, 6)
# 15
# 
# _Pozor:_ Ni nujno, da se $a$ in $b$ pojavita v drevesu.
##################################################################000516@#

    def vsota_interval(self,a,b):
        d=self
        if (not d.prazno):
            a=0
            b=0
            try:
                a=IskalnoDrevo.vsota_interval(d.levo,a,b)
            except AttributeError:
                pass   
            try:
                b=IskalnoDrevo.vsota_interval(d.desno,a,b)
            except AttributeError: 
                pass  
            if (d.vsebina <= b and a <= d.vsebina):
                return (int(d.vsebina)+a+b)
            else:
                return (a+b)
            
        else:
            return 0









































































































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
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).prestej_manjse(5)""", 4),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).prestej_manjse(1)""", 0),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).prestej_manjse(3)""", 2),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).prestej_manjse(8)""", 6),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).prestej_manjse(9)""", 7),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).prestej_manjse(10)""", 8),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).prestej_manjse(113)""", 8),    
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(-13)""", 0),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(0)""", 0),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(1)""", 0),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(3)""", 2),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(7)""", 6),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(8)""", 7),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(10)""", 9),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(11)""", 9),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(12)""", 9),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(13)""", 10),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).prestej_manjse(734)""", 10),
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
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(2, 6)""", 15),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(-22, 1)""", 1),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(-22, -5)""", 0),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(-2, 50)""", 40),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(4, 4)""", 4),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(7, 7)""", 7),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(5, 5)""", 0),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(7, 500)""", 24),
                ("""IskalnoDrevo([3, 9, 2, 4, 1, 8, 7, 6]).vsota_interval(4000, 5000)""", 0),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).vsota_interval(2, 6)""", 20),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).vsota_interval(-500, 3)""", 6),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).vsota_interval(3, 7)""", 25),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).vsota_interval(5, 5)""", 5),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).vsota_interval(10, 11)""", 0),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).vsota_interval(11, 15)""", 12),
                ("""IskalnoDrevo([3, 9, 5, 2, 4, 1, 8, 7, 6, 12]).vsota_interval(-5000, 5000)""", 57),
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
