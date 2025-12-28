import sys
import os
import threading
from pathlib import Path
from typing import Optional, Tuple, List

# Global lock to prevent race conditions on OneLoop.py
SVM_RANKER_LOCK = threading.Lock()

class SVMRankerClient:
    def __init__(self, tool_root: str | Path):
        self.tool_root = Path(tool_root).resolve()
        self.src_dir = self.tool_root / "src"
        
        # Add src directory to sys.path to allow imports
        if str(self.src_dir) not in sys.path:
            sys.path.append(str(self.src_dir))
            
        self._modules_loaded = False
        self._load_modules()

    def _load_modules(self):
        """Attempt to import SVMRanker modules."""
        try:
            # Delayed import to avoid errors if paths are not set up
            global parseBoogieProgramMulti, SVMLearnMulti
            # Note: These modules are expected to be in the tool_root/src directory
            from BoogieParser import parseBoogieProgramMulti
            from SVMLearn import SVMLearnMulti
            self._modules_loaded = True
        except ImportError as e:
            print(f"[Error] Failed to import SVMRanker modules: {e}")
            self._modules_loaded = False

    def run(self, file_path: Path, mode: str = "lnested", depth: int = 1) -> Tuple[str, Optional[str], List[str]]:
        """
        Run SVMRanker on the given file.
        
        Args:
            file_path: Path to the file (C or Boogie).
            mode: "lnested" or "lmulti".
            depth: Depth bound.
            
        Returns:
            Tuple containing:
            - result_status: "TERMINATE", "NONTERM", "UNKNOWN", or "ERROR"
            - ranking_function: The first valid ranking function string (if any)
            - rf_list: List of all generated ranking functions
        """
        if not self._modules_loaded:
            return "ERROR", None, []

        abs_file_path = file_path.resolve()
        
        # Configuration based on guide
        sample_strategy = "ENLARGE"
        cutting_strategy = "MINI"
        template_strategy = "SINGLEFULL"
        print_level = 0  # Suppress internal printing

        with SVM_RANKER_LOCK:
            try:
                # Switch cwd to SVMRanker/src because it generates temporary files there
                old_cwd = os.getcwd()
                os.chdir(self.src_dir)
                
                try:
                    # Step A: Parse
                    # The guide says parseBoogieProgramMulti takes a .bpl file.
                    # If the user says C is supported, we assume this parser handles it 
                    # or we are passing a .bpl file converted elsewhere.
                    # For now, we pass the file path as is.
                    (sourceFilePath, sourceFileName, 
                     templatePath, templateFileName, 
                     Info, 
                     parse_oldtime, parse_newtime) = parseBoogieProgramMulti(str(abs_file_path), "OneLoop.py")

                    # Step B: Learn
                    result, rf_list = SVMLearnMulti(
                        sourceFilePath, 
                        sourceFileName, 
                        depth,
                        parse_oldtime, 
                        parse_newtime, 
                        sample_strategy, 
                        cutting_strategy, 
                        template_strategy, 
                        print_level
                    )
                    
                    first_rf = rf_list[0] if rf_list else None
                    return result, first_rf, rf_list

                finally:
                    # Restore cwd
                    os.chdir(old_cwd)

            except Exception as e:
                print(f"[Error] SVMRanker execution exception: {e}")
                return "ERROR", None, []
