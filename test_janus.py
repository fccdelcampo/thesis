import janus_swi as janus
try:
    code = "hello(world)."
    print("Testing janus.consult('user', code)")
    janus.consult("user", code)
    print("Success 1!")
except Exception as e:
    print(f"Error 1: {e}")

try:
    print("Testing janus.consult('user', data=code)")
    janus.consult("user", data=code)
    print("Success 2!")
except Exception as e:
    print(f"Error 2: {e}")
