import janus_swi as janus

janus.query_once("op(900, fy, '~')")
janus.query_once("op(900, xfx, '⊃')")
janus.query_once("op(900, xfx, '∨')")
janus.query_once("op(900, xfx, '∧')")
janus.query_once("op(900, xfx, '=>')")

db = "if_then(A, B) :- or(A, _), ~B.\nformula( (G ∨ H) ⊃ ~I )."
try:
    janus.consult('user', db)
    print('Consult finished successfully.')
except Exception as e:
    print('Failed:', e)
