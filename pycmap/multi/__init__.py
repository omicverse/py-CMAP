"""pycmap.multi — cross-modal spatial mapping (scATAC-seq -> spatial).

Uses GLUE guidance graph embeddings to bridge modality gap,
then CMAP's three-level pipeline for precise spatial assignment.

Requires the optional extra::

    pip install "pycmap[multi]"

which pulls in scglue, torch, networkx and matplotlib. Importing this
subpackage without them raises ImportError; ``pycmap`` itself does not
import it, so the core package stays installable without a GPU stack.
"""

__version__ = "0.1.0"

from .glue_bridge import run_cmap_atac
