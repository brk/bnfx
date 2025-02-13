# bnfx
Grammar slicing and dicing

In short: it's ANTLR-style EBNF, with re2c-style token definitions.

The intent is to have a single grammar definition which can export to
a set of targets that I care about, primarily:
  ANTLRv3 (for interactive experimentation/debugging),
  Lemon (for fast parsing),
  tree-sitter (for editor support).

There is also support for generating railroad diagrams.
Currently text only. Eventually I'll add an HTML-embedding-SVGs mode.

Currently, bnfx files do not support embedded action code.
I haven't decided whether to support arbitrary action code
or what.

There is a rough draft of generating recursive descent
recognizer Rust code (which assumes the grammar is LL(1)).
Generally, those sorts of assumptions are not checked,
since it is anticipated that external tools will be doing the checking.
As is, bnfx is not meant to be a standalone one-stop-shop sort of tool.

## Usage

See `grammars/` directory for some examples of bnfx syntax.


```
$ cat grammars/json.bnfx
source_file: value EOF;
value
  : object
  | array
  | STR
  | NUM
  | TRU
  | FLS
  | NUL
  ;

object: LCURLY members? RCURLY;
members: member (COMMA member)*;
member: STR COLON value;
array: LBRACKET elements? RBRACKET;
elements: value (COMMA value)*;

LCURLY = "{";
RCURLY = "}";
COMMA = ",";
COLON = ":";
LBRACKET = "[";
RBRACKET = "]";
TRU = "true";
FLS = "false";
NUL = "null";
NUM = [0-9]+ floatpart?;
floatpart = "." [0-9]+ ;
STR = ["] [^"]* ["] ;
```

Grammars can be printed in multiple formats.
Note that when targeting the LALR-based Lemon parser,
EBNF syntax is eliminated and left recursion is used.
When targeting `grammophone`, right recursion is used
to avoid introducing artificial LL(1) conflicts.

```
$ uv run bnfx.py grammars/json.bnfx --to lemon
source_file ::= value EOF.

value ::= object.
value ::= array.
value ::= STR.
value ::= NUM.
value ::= TRU.
value ::= FLS.
value ::= NUL.

object ::= LCURLY members_opt RCURLY.

members ::= member _COMMA_member_star.

member ::= STR COLON value.

array ::= LBRACKET elements_opt RBRACKET.

elements ::= value _COMMA_value_star.

members_opt ::= members.
members_opt ::= .

_COMMA_member ::= COMMA member.

_COMMA_member_star ::= _COMMA_member_star _COMMA_member.
_COMMA_member_star ::= .

elements_opt ::= elements.
elements_opt ::= .

_COMMA_value ::= COMMA value.

_COMMA_value_star ::= _COMMA_value_star _COMMA_value.
_COMMA_value_star ::= .
```


```
$ uv run bnfx.py grammars/json.bnfx --to antlr
source_file : value EOF;

value : object | array | STR | NUM | TRU | FLS | NUL;

object : LCURLY members? RCURLY;

members : member (COMMA member)*;

member : STR COLON value;

array : LBRACKET elements? RBRACKET;

elements : value (COMMA value)*;
```

For tree-sitter only, the output includes (processed) token rules:

```
$ uv run bnfx.py grammars/json.bnfx --to treesitter
source_file : ($) => seq($.value, $.EOF),

value : ($) => choice($.object, $.array, $.STR, $.NUM, $.TRU, $.FLS, $.NUL),

object : ($) => seq($.LCURLY, optional($.members), $.RCURLY),

members : ($) => seq($.member, repeat(seq($.COMMA, $.member))),

member : ($) => seq($.STR, $.COLON, $.value),

array : ($) => seq($.LBRACKET, optional($.elements), $.RBRACKET),

elements : ($) => seq($.value, repeat(seq($.COMMA, $.value))),


LCURLY : $ => "{",
RCURLY : $ => "}",
COMMA : $ => ",",
COLON : $ => ":",
LBRACKET : $ => "[",
RBRACKET : $ => "]",
TRU : $ => "true",
FLS : $ => "false",
NUL : $ => "null",
NUM : $ => /[0-9]+([.][0-9]+)?/,
STR : $ => /["][^"]*["]/,

```


```
$ uv run bnfx.py grammars/json.bnfx --to rrtext
source_file ::=
         ┌───────┐     ╭─────╮        
├┼───────│ value │─────│ EOF │──────┼┤
         └───────┘     ╰─────╯        

value ::=
         ┌────────┐        
├┼──╮────│ object │───╭──┼┤
    │    └────────┘   │    
    │                 │    
    │    ┌───────┐    │    
    ╰────│ array │────╯    
    │    └───────┘    │    
    │                 │    
    │     ╭─────╮     │    
    ╰─────│ STR │─────╯    
    │     ╰─────╯     │    
    │                 │    
    │     ╭─────╮     │    
    ╰─────│ NUM │─────╯    
    │     ╰─────╯     │    
    │                 │    
    │     ╭─────╮     │    
    ╰─────│ TRU │─────╯    
    │     ╰─────╯     │    
    │                 │    
    │     ╭─────╮     │    
    ╰─────│ FLS │─────╯    
    │     ╰─────╯     │    
    │                 │    
    │     ╭─────╮     │    
    ╰─────│ NUL │─────╯    
          ╰─────╯          

object ::=
                      ╭───────────────╮                     
                      │               │                     
         ╭────────╮   │  ┌─────────┐  │   ╭────────╮        
├┼───────│ LCURLY │───╯──│ members │──╰───│ RCURLY │──────┼┤
         ╰────────╯      └─────────┘      ╰────────╯        

members ::=
                      ╭───────────────────────────────────────╮      
                      │                                       │      
         ┌────────┐   │        ╭───────╮     ┌────────┐       │      
├┼───────│ member │───╯─╭──────│ COMMA │─────│ member │─────╮─╰────┼┤
         └────────┘     │      ╰───────╯     └────────┘     │        
                        ╰───────────────────────────────────╯        

member ::=
         ╭─────╮     ╭───────╮     ┌───────┐        
├┼───────│ STR │─────│ COLON │─────│ value │──────┼┤
         ╰─────╯     ╰───────╯     └───────┘        

array ::=
                        ╭────────────────╮                       
                        │                │                       
         ╭──────────╮   │  ┌──────────┐  │   ╭──────────╮        
├┼───────│ LBRACKET │───╯──│ elements │──╰───│ RBRACKET │──────┼┤
         ╰──────────╯      └──────────┘      ╰──────────╯        

elements ::=
                     ╭──────────────────────────────────────╮      
                     │                                      │      
         ┌───────┐   │        ╭───────╮     ┌───────┐       │      
├┼───────│ value │───╯─╭──────│ COMMA │─────│ value │─────╮─╰────┼┤
         └───────┘     │      ╰───────╯     └───────┘     │        
                       ╰──────────────────────────────────╯        
```

## Future Work

- Check LL(k) for k > 1
- Handle rule elt labels in non-trivial ways
- Permit control over whether optionals are expanded with extra rules or with extra alts.


