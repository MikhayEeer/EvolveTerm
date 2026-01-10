// Source: data/benchmarks/code2inv/1.c
#include <stdlib.h>
#include "seahorn/seahorn.h"
#define assume(e) if(!(e)) exit(-1);

int main() {
  
  int x;
  int y;
  
  (x = 1);
  (y = 0);
  
  while ((y < 100000)) {
      sassert(y >= y0);

    {
    (x  = (x + y));
    (y  = (y + 1));
    }

  }
  
{;
//@ assert( (x >= y) );
}

}