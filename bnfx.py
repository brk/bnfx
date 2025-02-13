import argparse
from sly import Lexer, Parser
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import enum
import pprint

class GrammarFlavor(enum.Enum):
    LEMON = 'lemon'
    ANTLR = 'antlr'
    GRAMMOPHONE = 'grammophone'
    TREESITTER = 'treesitter'
    DICT = 'dict'
    REPR = 'repr'

type RuleRef = str
type TokenRef = str
type MixedRef = str
type ActionVar = str
type LabelOp = str

@dataclass
class AtomTerminal:
    tokref: TokenRef

    def render(self, flavor):
        if flavor == GrammarFlavor.TREESITTER:
            return f"$.{self.tokref}"
        return self.tokref

@dataclass
class AtomNonterm:
    rulref: RuleRef

    def render(self, flavor):
        if flavor == GrammarFlavor.TREESITTER:
            return f"$.{self.rulref}"
        return self.rulref

@dataclass
class AtomBlock:
    alts: List["Alt"]

    def render(self, flavor):
        if flavor == GrammarFlavor.TREESITTER:
            return render_treesitter_alts(self.alts)
        return '(' + '|'.join(alt.render(flavor) for alt in self.alts) + ')'

type Atom = AtomTerminal | AtomNonterm | AtomBlock
type AtomRef = AtomTerminal | AtomNonterm

def atom_name(ar: AtomRef) -> str:
    match ar:
        case AtomNonterm(rulref):
            return rulref
        case AtomTerminal(tokref):
            return tokref

def lemon_suffix(enbfsuffix: str) -> str:
    return { "*": "_star",
             "+": "_plus",
             "?": "_opt",
             "" : ""
            }[enbfsuffix]

def treesitter_render_atom_suffix(rendered_atom, suffix):
    base = rendered_atom

    match suffix:
        case "*":
            return f"repeat({base})"
        case "+":
            return f"repeat1({base})"
        case "?":
            return f"optional({base})"
        case _:
            return base

@dataclass
class Elt:
    label: Optional[Tuple[ActionVar, LabelOp]]
    atom: Atom
    suffix: str

    def match_bare_terminal(self) -> Optional[str]:
        if self.suffix != "":
            return None
        match self.atom:
            case AtomTerminal(tokref):
                return tokref
            case _:
                return None

    def match_bare_nonterminal(self) -> Optional[str]:
        if self.suffix != "":
            return None
        match self.atom:
            case AtomNonterm(rulref):
                return rulref
            case _:
                return None

    def render(self, flavor: GrammarFlavor) -> str:
        if flavor == GrammarFlavor.ANTLR:
            if self.label is None:
                prefix = ""
            else:
                prefix = ''.join(self.label)
            return prefix + self.atom.render(flavor) + self.suffix

        if flavor == GrammarFlavor.LEMON:
            suffix = str(self.suffix)

            if self.label is not None:
                suffix += f"({self.label[0]})"

            return self.atom.render(flavor) + suffix

        if flavor == GrammarFlavor.REPR:
            return repr(self)

        if flavor == GrammarFlavor.TREESITTER:
            r = self.atom.render(flavor)
            base = treesitter_render_atom_suffix(r, self.suffix)
            if self.label is None:
                return base
            return f"field('{self.label[0]}', {base})"

        if flavor == GrammarFlavor.DICT:
            return str(self)

        raise ValueError(flavor)

    def trivially_nullable(self):
        if self.suffix in ['*', '?']:
            return True
        match self.atom:
            case AtomBlock(alts):
                return len(alts) == 0
            case AtomNonterm(rulref):
                return rulref.endswith("_star") or rulref.endswith("_opt")
            case _:
                return False

    def firsts_shallow(self, explicit_epsilons=True) -> List[Optional[MixedRef]]:
        match self.atom:
            case AtomBlock(alts):
                assert len(alts) > 0
                base = alts[0].firsts_shallow()
            case AtomTerminal(tokref):
                base = [tokref]
            case AtomNonterm(rulref):
                base = [rulref]

        if explicit_epsilons and self.trivially_nullable():
            return base + [None]
        return base

@dataclass
class Alt:
    elts: List[Elt]

    def render(self, flavor: str) -> str:
        if flavor == GrammarFlavor.TREESITTER:
            elts = [elt.render(flavor) for elt in self.elts]
            match elts:
                case [rendered]:
                    return rendered
                case _:
                    parts = ', '.join(elts)
                    return f"seq({parts})"
        return ' '.join(self.render_list(flavor))

    def render_list(self, flavor: str) -> List[str]:
        return [elt.render(flavor) for elt in self.elts]

    def firsts_shallow(self) -> List[Optional[MixedRef]]:
        if not self.elts:
            return [None]

        firsts = set()
        for elt in self.elts:
            efs = elt.firsts_shallow(explicit_epsilons=False)
            firsts.update(set(efs))
            if not elt.trivially_nullable():
                break
        if not firsts:
            return [None]
        return list(firsts)

# Lexical rules closely mirror the structure of EBNF rules
# but with a slightly different notion of atoms; also,
# lexical rules must be acyclic. (Although that is currently
# assumed without checking).

@dataclass
class RegexNamed:
    name: str

@dataclass
class RegexLiteral:
    value: str

@dataclass
class RegexClass:
    contents: str

type RegexAtom = RegexNamed | RegexLiteral | RegexClass

@dataclass
class RegexElt:
    base: "Regex"
    suffix: str

@dataclass
class RegexSeq:
    elts: List[RegexElt | RegexAtom]

@dataclass
class RegexAlts:
    alts: List[RegexElt | RegexAtom]


type Regex = RegexSeq | RegexAlts | RegexElt | RegexAtom

def neutralize(c: str) -> str:
    """Special characters (such as .) must not appear
    un-quoted within a regex. Special characters are
    interpreted literally when they appear within a
    regex range bracket."""

    if c in "[]" or c.isalnum():
        return c
    return f"[{c}]"


def Regex_render(r: Regex, flavor: GrammarFlavor, depth: int) -> str:
    assert r is not None
    match r:
        case RegexNamed(name):
            if flavor == GrammarFlavor.TREESITTER:
                raise Exception("should have been eliminated")
            raise Exception("not yet implemented")
        case RegexLiteral():
            if flavor == GrammarFlavor.TREESITTER:
                if depth == 0:
                    return '"' + r.value + '"'
                return ''.join(neutralize(c) for c in r.value)
            raise Exception("not yet implemented")
        case RegexClass():
            return "[" + r.contents + "]"
        case RegexElt():
            if flavor == GrammarFlavor.TREESITTER:
                return Regex_render(r.base, flavor, depth + 1) + r.suffix
            raise Exception("not yet implemented")
        case RegexSeq():
            if flavor == GrammarFlavor.TREESITTER:
                parts = "".join(Regex_render(e, flavor, depth + 1) for e in r.elts)
                if depth == 0:
                    return parts
                return f"({parts})"
            raise Exception("not yet implemented")
        case RegexAlts():
            if flavor == GrammarFlavor.TREESITTER:
                return "|".join(Regex_render(e, flavor, depth) for e in r.alts)
            raise Exception("not yet implemented")
        case _:
            raise Exception("unhandled Regex_render case: " + str(r))

@dataclass
class LexicalRule:
    name: str
    alts: List[Regex]

    def render(self, flavor: GrammarFlavor) -> str:
        if len(self.alts) == 1:
            a = self.alts[0]
            if isinstance(self.alts[0], RegexLiteral):
                quoted = Regex_render(a, flavor, depth=0)
                if quoted.count('"') == 2:
                    return quoted

        alts = Regex_render(RegexAlts(self.alts), flavor, depth=0)
        return f"/{alts}/"

def cook_lex(rules: List[LexicalRule]) -> List[LexicalRule]:
    """Given a set of lexical rules, eliminates all named references
    by replacing them with their corresponding definitions."""

    byname = {r.name: r for r in rules}
    def replace_r(r):
        match r:
            case RegexNamed(name):
                x = byname[name]
                match x:
                    case LexicalRule():
                        lexrule = replace_all(byname[name])
                        y = RegexAlts(lexrule.alts)
                        byname[name] = y
                        return y
                    case _:
                        return x
            case RegexLiteral():
                return r
            case RegexClass():
                return r
            case RegexElt():
                return RegexElt(replace_r(r.base), r.suffix)
            case RegexSeq():
                return RegexSeq([replace_r(e) for e in r.elts])

    def replace_all(r: LexicalRule) -> LexicalRule:
        return LexicalRule(r.name, [replace_r(g) for g in r.alts])

    return [replace_all(r) for r in rules]

@dataclass
class Rule:
    name: str
    alts: List[Alt]

    def render_alts(self, flavor: GrammarFlavor) -> str:
        if flavor == GrammarFlavor.TREESITTER:
            return render_treesitter_alts(self.alts)
        return ' | '.join(alt.render(flavor) for alt in self.alts)

def Rule_nested_lists_via(r: Rule, flavor: str) -> List[List[str]]:
    return [alt.render_list(flavor) for alt in r.alts]

def render_treesitter_alts(altlist: List[Alt]) -> str:
    alts = [alt.render(GrammarFlavor.TREESITTER) for alt in altlist]
    match alts:
        case [rendered]:
            return rendered
        case _:
            choices = ', '.join(alts)
            return f"choice({choices})"

def atom_firsts(v, firsts):
    match v:
        case AtomTerminal():
            return {v.tokref}
        case AtomNonterm():
            return firsts[v.rulref]
        case _:
            raise Exception("atom block not yet supported")

def cg_match(arms) -> str:
    syntactically_exhaustive = any(arm.startswith("_") for arm in arms)
    if not syntactically_exhaustive:
        arms.append("_ => { Err(()) }")
    # TODO: distinguish exhaustive vs inexhaustive matches, rather than
    # forcibly making matches syntactically exhaustive.
    arms = ",\n".join(arms)
    return f"""
match self.peek_token() {{
{arms}
}}
"""


def cg(b_x: "Bnfx") -> str:
    bnfx = b_x.normalize_bnf(left_recursion=False)
    firsts = bnfx.firsts()
    def firsts_(v) -> Set[Optional[str]]:
        match v:
            case AtomTerminal():
                return atom_firsts(v, firsts)
            case AtomNonterm():
                return atom_firsts(v, firsts)
            case Alt():
                if len(v.elts) == 0:
                    return "_"
                if len(v.elts) == 1 and v.elts[0].suffix == '':
                    return atom_firsts(v.elts[0].atom, firsts)

                raise Exception("first_ of " + str(v))
            case _:
                raise Exception("first_ of " + str(v))
    def match_arm(firsts, mb_x, act):
        if None in firsts:
            pat = "_"
        else:
            pat = "|".join(f for f in firsts)

        # The construction we have here is simple, but tends to
        # generate code which inspects tokens twice: first a
        # rule will peek at a token and compare it against the firsts
        # to dispatch it to the right alt, then the alt rule will
        # eat the token, which is a second peek-and-compare.
        #
        # We can instead use a scheme in which most such constructs
        # would consume a token, then dispatch to a rule variant which
        # is, Brzozowski-style, the derivative w/r/t the known token.
        #
        # Part of the code looks like this:
        #      if len(firsts) == 1 and mb_x is not None:
        #          match mb_x:
        #              case Alt(elts=[elt]):
        #                  nt = elt.match_bare_nonterminal()
        #                  if nt:
        #                      act = act.replace(nt, nt + "_after_" + firsts[0], 1)
        return f"{pat} => {{ {act} }}"
    def q_from(seq):
        if len(seq) == 0:
            return ""
        if len(seq) == 1:
            return q(seq[0])

        x, xs = seq[0], seq[1:]
        return q(x) + "?;\n" + q_from(xs)
    def q(v):
        match v:
            case AtomTerminal():
                return f"self.eat_token({v.tokref})"
            case AtomNonterm():
                return f"self.recognize_{v.rulref}()"
            case AtomBlock():
                return "AtomBlock..."
            case Elt():
                match v.suffix:
                    case "":
                        return q(v.atom)
                    case "+":
                        return str(v)
                    case "?":
                        return str(v)
                    case "*":
                        match v.atom:
                            case AtomBlock():
                                if len(v.atom.alts) == 1:
                                    alt = v.atom.alts[0]
                                    if len(alt.elts) > 1:
                                        t = alt.elts[0].match_bare_terminal()
                                        if t:
                                            return f"""
    while self.eat_token({t}) {{
            {q(Alt(alt.elts[1:]))}
    }}"""
                                    return str(v)
                            case _:
                                return str(v)
            case Alt():
                if v.elts == []:
                    return "Ok(())"

                return q_from(v.elts)
            case Rule():
                if len(v.alts) == 1:
                    code = q(v.alts[0])
                elif v.name.endswith("_star"):
                    rootname = v.name[:-5]
                    lookfor = firsts[rootname]
                    code = ("loop { todo: match looking for " + str(lookfor)
                            + " else break ... }")
                    match list(lookfor):
                        case [lookfor]:
                            prefix = "_" + lookfor + "_"
                            if rootname.startswith(prefix):
                                contpart = rootname[len(prefix):]
                                code = f"""
loop {{
  if self.peek_token() != {lookfor} {{ break; }}
  self.recognize_{contpart}();
}}
"""
                else:
                    code = cg_match(list(match_arm(list(firsts_(x)), x, q(x))
                                         for x in v.alts))

                primary = f"""
fn recognize_{v.name}(&mut self) -> Result<(),()> {{
{code}
}}
"""
                if False:
                    match list(firsts[v.name]):
                        case [lookfor]:
                            secondary = f"""
    fn recognize_{v.name}_after_{lookfor}(&mut self) -> Result<(),()> {{
    {code}
    }}
    """.replace(f"self.eat_token({lookfor})?;", "", 1)
                            fndefs = primary + "\n" + secondary
                        case _:
                            fndefs = primary
                else:
                    fndefs = primary

                return fndefs
    comments = """
// This code assumes that the terminal names are all i32 constants
// that have been brought into scope.
//
// The generated rules below are intended to be used within
// a context like the following (but modified to suit your
// own needs):

/**********************************************************
struct TokIter {
    toks: Vec<YourTokenType>,
    idx: usize,
}

impl TokIter {
    fn next_token(&mut self) -> i32 {
        let rv = self.peek_token();
        self.idx += 1;
        rv
    }

    fn peek_token(&self) -> i32 {
        self.toks[self.idx].tok
    }

    fn eat_token(&mut self, t: i32) -> Result<(), ()> {
        let x = self.next_token();
        if x == t {
            Ok(())
        } else {
            println!("expected {}, got {}", raw_token_name(t), raw_token_name(x));
            Err(())
        }
    }

...

}
**********************************************************/
    """
    return comments + "\n\n".join(q(v) for v in bnfx.rules)


@dataclass
class BnfxDict:
    start: RuleRef
    rules: dict

@dataclass
class Bnfx:
    rules: List[Rule] # only one Rule per .name
    lexerrules: List[LexicalRule]

    def rule(self, name: str) -> Rule:
        return [r for r in self.rules if r.name == name][0]

    def to_dict(self, flavor=GrammarFlavor.REPR) -> BnfxDict:
        start = self.rules[0].name
        rules = {r.name: [] for r in self.rules}
        for r in self.rules:
            rules[r.name].extend(Rule_nested_lists_via(r, flavor))
        return BnfxDict(start, rules)

    def render(self, flavor=GrammarFlavor.DICT) -> str:
        if flavor == GrammarFlavor.REPR:
            return pprint.pformat(self, indent=2)

        if flavor == GrammarFlavor.DICT:
            return pprint.pformat(self.to_dict(GrammarFlavor.DICT), indent=2)

        if flavor == GrammarFlavor.LEMON:
            return render_bnf(self, rulesep="::=", left_recursion=True)

        if flavor == GrammarFlavor.GRAMMOPHONE:
            return render_bnf(self, rulesep="->", left_recursion=False)

        if flavor == GrammarFlavor.ANTLR:
            lines = []
            for r in self.rules:
                lines.append(r.name + " : " + r.render_alts(flavor) + ";")
                lines.append("")
            return "\n".join(lines)

        if flavor == GrammarFlavor.TREESITTER:
            lines = []
            for r in self.rules:
                lines.append(r.name + " : ($) => " + r.render_alts(flavor) + ",")
                lines.append("")

            lines.append("")
            for r in cook_lex(self.lexerrules):
                if r.name[0].isupper():
                    lines.append(r.name + " : $ => " + r.render(flavor) + ",")
            return "\n".join(lines)

        raise ValueError(flavor)

    def nonterminals(self) -> Set[RuleRef]:
        return set(r.name for r in self.rules)

    def terminals(self) -> Set[TokenRef]:
        ts = set()
        def consider_alt(alt):
            for elt in alt.elts:
                match elt.atom:
                    case AtomBlock(alts):
                        for alt in alts:
                            consider_alt(alt)
                    case AtomTerminal(tokref):
                        ts.add(tokref)
                    case AtomNonterm():
                        pass
                    case other:
                        raise ValueError(other)

        for r in self.rules:
            for alt in r.alts:
                consider_alt(alt)
        return ts

    def firsts(self) -> dict:
        nonterms = self.nonterminals()
        first = {nt: set() for nt in nonterms}
        def nt_first(nt):
            if first[nt]:
                return first[nt]
            r = self.rule(nt)
            for alt in r.alts:
                for f in alt.firsts_shallow():
                    if f == nt:
                        continue

                    if f is None:
                        first[nt].add(None)
                    elif f in nonterms:
                        first[nt].update(nt_first(f))
                    else:
                        first[nt].add(f)
            return first[nt]

        for nt in self.nonterminals():
            nt_first(nt)

        return first

    def normalize_antlr() -> "Bnfx":
        """Really this means normalize for LL(1) structure.
            Rules marked as inlineable (by a preceding underscore)
            should be inlined."""
        raise Exception("not yet implemented")

    def normalize_bnf(self, left_recursion=True) -> "Bnfx":
        """Lemon does not support parenthesized blocks, so such occurrences
            must be transformed into separate rules."""
        # Separate newly-generated rules to ensure we don't change the
        # implicit start symbol.
        genrules = []
        samerules = []
        newrulenames = set()
        def blockname(alts: List[Alt]) -> str:
            altstrs = ['_'.join(elt.render(GrammarFlavor.LEMON)
                               for elt in alt.elts)
                               for alt in alts]
            # prefix underscore because this is an implictly inlinable rule
            return "_" + '_or_'.join(altstrs)

        def norm_elt(elt: Elt):
            if not elt.suffix:
                return elt

            basename = atom_name(elt.atom)
            rulename = basename + lemon_suffix(elt.suffix)
            ruleelt = Elt(label=None, atom=AtomNonterm(rulename), suffix="")
            eltbare = Elt(label=elt.label, atom=elt.atom, suffix="")
            # Since lemon is LALR the rules below are left-recursive.
            match elt.suffix:
                case "?":
                    alts = [Alt(elts=[eltbare]), Alt(elts=[])]
                    genrules.append(Rule(rulename, alts))
                case "*":
                    if left_recursion:
                        alts = [Alt(elts=[ruleelt, eltbare]), Alt(elts=[])]
                    else:
                        alts = [Alt(elts=[eltbare, ruleelt]), Alt(elts=[])]

                    genrules.append(Rule(rulename, alts))
                case "+":
                    if left_recursion:
                        alts = [Alt(elts=[ruleelt, eltbare]), Alt(elts=[eltbare])]
                    else:
                        alts = [Alt(elts=[eltbare, ruleelt]), Alt(elts=[eltbare])]
                    genrules.append(Rule(rulename, alts))
            return ruleelt

        def norm_alts(alts):
            newalts = []
            for alt in alts:
                newelts = []
                for elt in alt.elts:
                    match elt.atom:
                        case AtomBlock(_):
                            blockalts = norm_alts(elt.atom.alts)
                            newrulename = blockname(blockalts)
                            newrulenames.add(newrulename)
                            genrules.append(Rule(newrulename, blockalts))
                            newelts.append(norm_elt(Elt(label=elt.label,
                                               atom=AtomNonterm(newrulename),
                                               suffix=elt.suffix)))
                        case _:
                            newelts.append(norm_elt(elt))
                newalts.append(Alt(newelts))
            return newalts

        for r in self.rules:
            alts = norm_alts(r.alts)
            samerules.append(Rule(r.name, alts))

        return BnfxRaw(samerules + genrules, self.lexerrules).cook()

def render_bnf(b: Bnfx, rulesep: str, left_recursion):
    norm = b.normalize_bnf(left_recursion)
    lines = []
    for r in norm.rules:
        for alt in r.alts:
            lines.append(r.name + f" {rulesep} " + alt.render(GrammarFlavor.LEMON) + ".")
        lines.append("")
    return "\n".join(lines)

@dataclass
class BnfxRaw:
    rules: List[Rule]
    lexerrules: List[LexicalRule]

    def cook(self) -> Bnfx:
        """BnfxRaw permits duplicate names in the rules list;
        Bnfx requires canonicalization/deduplication."""

        d = {r.name: [] for r in self.rules}
        for r in self.rules:
            d[r.name].extend(r.alts)
        cooked = [Rule(n, alts) for n, alts in d.items()]
        assert cooked[0].name == self.rules[0].name
        return Bnfx(rules=cooked, lexerrules=self.lexerrules)


class BnfxLexer(Lexer):
    tokens = { TOKREF, RULREF, EBNFSUFFIX , LABELOP, VBAR, REGEX_CLASS, REGEX_DQUO }
    ignore = ' \t\n'
    literals = { ':', ';', '(', ')', '"', '=' }

    RULREF = r'_?[a-z][a-zA-Z0-9_]*'
    TOKREF = r'_?[A-Z][a-zA-Z0-9_]*'
    EBNFSUFFIX = r'[?+*]'
    LABELOP = r'[+]?#'
    VBAR = r'[|]'

    REGEX_CLASS = r'[\[][^\]]+[]]'
    REGEX_DQUO= r'"[^"]+"'

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1

def untuple(xs, idx=0):
    return [x[idx] for x in xs]

class BnfxParser(Parser):
    tokens = BnfxLexer.tokens

    def __init__(self):
        pass

    @_('rule { rule }')
    def grammarDef(self, p):
        mixedrules = [p.rule0] + untuple(p[1])
        parserrules = [r for r in mixedrules if isinstance(r, Rule)]
        lexerrules = [r for r in mixedrules if isinstance(r, LexicalRule)]
        return BnfxRaw(parserrules, lexerrules)

    @_("RULREF ':' altList ';'")
    def rule(self, p):
        return Rule(name=p.RULREF, alts=p.altList)

    @_("RULREF '=' regex ';'")
    def rule(self, p):
        return LexicalRule(p.RULREF, p.regex)

    @_("TOKREF '=' regex ';'")
    def rule(self, p):
        return LexicalRule(p.TOKREF, p.regex)

    @_("alt { VBAR alt }")
    def altList(self, p):
        return [p.alt0] + untuple(p[1], idx=1)

    @_('{ elt }')
    def alt(self, p):
        # sly wraps each Elt in a one-element tuple,
        # strange but this is the easiest fix for now.
        return Alt(elts=untuple(p[0]))

    @_('atom EBNFSUFFIX labelOp RULREF')
    def elt(self, p):
        label = (p.RULREF, p.labelOp)
        return Elt(label=label, atom=p.atom, suffix=p.ENBFSUFFIX)

    @_('atom labelOp RULREF')
    def elt(self, p):
        label = (p.RULREF, p.labelOp)
        return Elt(label=label, atom=p.atom, suffix='')

    @_('atom')
    def elt(self, p):
        return Elt(label=None, atom=p.atom, suffix='')

    @_('atom EBNFSUFFIX')
    def elt(self, p):
        return Elt(label=None, atom=p.atom, suffix=p.EBNFSUFFIX)

    @_("LABELOP")
    def labelOp(self, p):
        return p.LABELOP

    @_('TOKREF')
    def atom(self, p):
        return AtomTerminal(p.TOKREF)

    @_('RULREF')
    def atom(self, p):
        return AtomNonterm(p.RULREF)

    @_("'(' altList ')'")
    def atom(self, p):
        return AtomBlock(p.altList)

    @_("regex_alt { VBAR regex_alt }")
    def regex(self, p):
        return [p.regex_alt0] + untuple(p[1], idx=1)

    @_("{ regex_elt }")
    def regex_alt(self, p):
        elts = untuple(p[0])
        if len(elts) == 1:
            return elts[0]
        return RegexSeq(elts)

    @_("regex_atom EBNFSUFFIX")
    def regex_elt(self, p):
        return RegexElt(p.regex_atom, p.EBNFSUFFIX)

    @_("regex_atom")
    def regex_elt(self, p):
        return p.regex_atom

    @_("REGEX_CLASS")
    def regex_atom(self, p):
        return RegexClass(p.REGEX_CLASS[1:-1])

    @_("REGEX_DQUO")
    def regex_atom(self, p):
        return RegexLiteral(p.REGEX_DQUO[1:-1])

    @_("RULREF")
    def regex_atom(self, p):
        return RegexNamed(p.RULREF)

    @_("'(' regex ')'")
    def regex_atom(self, p):
        return p.regex


def Diagram_for_Rule(r: Rule, rulename_as_comment=True):
    import railroad

    def Atom_item(a: Atom):
        match a:
            case AtomTerminal():
                assert a.tokref is not None
                return railroad.Terminal(a.tokref)
            case AtomNonterm():
                return railroad.NonTerminal(a.rulref)
            case AtomBlock():
                return Alts(a.alts)

    def Elt_unlabeled_item(e: Elt):
        item = Atom_item(e.atom)
        match e.suffix:
            case "?":
                return railroad.Optional(item, skip=False)
            case "*":
                return railroad.ZeroOrMore(item, repeat=None, skip=False)
            case "+":
                return railroad.OneOrMore(item, repeat=None, skip=False)
            case "":
                return item

    def Elt_item(e: Elt):
        assert e is not None
        item = Elt_unlabeled_item(e)
        if e.label is None or e.label == "":
            return item
        else:
            return railroad.Group(item, e.label[0])

    def Alt_item(a: Alt):
        items = [Elt_item(e) for e in a.elts]
        return railroad.Sequence(*items)

    def Alts(alts: List[Alt]):
        items = [Alt_item(a) for a in alts]
        return railroad.Choice(0, *items)

    if rulename_as_comment:
        return railroad.Diagram(railroad.Comment(r.name), Alts(r.alts))
    else:
        return railroad.Diagram(Alts(r.alts))

def Rr_print_text(b: Bnfx):
    import sys

    for r in b.rules:
        d = Diagram_for_Rule(r, rulename_as_comment=False)
        print(r.name, "::=")
        d.writeText(sys.stdout.write)
        print()



def example_ebnf_grammar():
    return """
        hello: WORLD? ITS? ME@a;
        foo: (bar BAZ)+ BUX;
        bar: ME | YOU ME?;
    """

def main():
    parser = argparse.ArgumentParser(description="Process a BNFX file.")
    parser.add_argument("bnfxpath", type=str, help="Path to the input file")
    parser.add_argument("--to", type=str, help="Grammar flavor (lemon, antlr, grammophone, treesitter), debug format (dict, repr, misc), railroad format (rrtext, rrhtml), codegen target (rs)", default="rrtext")

    # Parse the arguments
    args = parser.parse_args()

    if not args.bnfxpath:
        print("Must supply a path to an input BNFX file.")
        print("Try using    grammars/json.bnfx")
        sys.exit(1)

    text = open(args.bnfxpath, 'r').read()

    lexer = BnfxLexer()
    parser = BnfxParser()

    rvraw = parser.parse(lexer.tokenize(text))
    rv = rvraw.cook()

    match args.to:
        case "antlr":
            print(rv.render(GrammarFlavor.ANTLR))
        case "lemon":
            print(rv.render(GrammarFlavor.LEMON))
        case "grammophone":
            print(rv.render(GrammarFlavor.GRAMMOPHONE))
        case "treesitter":
            print(rv.render(GrammarFlavor.TREESITTER))
        case "dict":
            pprint.pprint(rv.to_dict(GrammarFlavor.ANTLR))
        case "repr":
            pprint.pprint(rv.to_dict(GrammarFlavor.REPR))
        case "raw": # undocumented
            pprint.pprint(rvraw)
        case "misc":
            pprint.pprint(rv)
            print("terminals: ", rv.terminals())
            print("nonterminals:", rv.nonterminals())
            print("firsts:")
            pprint.pprint(rv.firsts(), indent=2)
        case "allgrammars":
            print("========================================")
            for flavor in GrammarFlavor:
                print("flavor:", flavor)
                print("~~~~~~~~~~~~~~~~~~~~")
                if flavor not in [GrammarFlavor.REPR,
                                  GrammarFlavor.DICT]:
                    print(rv.render(flavor))
                print("----------------------------------------")
            print("========================================")
        case "rs":
            print(cg(rv))
        case "rrhtml":
            raise Exception("not yet implemented")
        case "rrtext":
            Rr_print_text(rv)
        case _:
            print("unrecognized target flavor:", args.to)


if __name__ == '__main__':
    main()

