import janus_swi as janus
import sys

print("Attempting to install 'scasp' pack for SWI-Prolog via Janus...")

try:
    # Check if already installed?
    # q = janus.query("pack_property(scasp, version(V))")
    # ... but let's just try install.
    
    # query_once returns the first solution or truthy.
    # pack_install(scasp, [interactive(false)])
    janus.query_once("pack_install(scasp, [interactive(false)])")
    print("SUCCESS: 'scasp' installed successfully.")
except Exception as e:
    print(f"FAILURE: Could not install 'scasp'. Error: {e}")
    sys.exit(1)
