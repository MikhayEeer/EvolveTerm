

int main()
{
    int a, b, q, olda;
    q = 0;
    a = 0;
    b = 0;
    while (q > 0) {
      q = q + a - 1;
      olda = a;
      a = 3*olda - 4*b;
      b = 4*olda + 3*b;
    }
    return 0;
}