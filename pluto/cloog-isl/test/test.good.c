/* Generated from ../../../git/cloog/test/test.cloog by CLooG 0.14.0-72-gefe2fc2 gmp bits in 0.02s. */
extern void hash(int);

/* Useful macros. */
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define max(x,y)    ((x) > (y) ? (x) : (y))
#define min(x,y)    ((x) < (y) ? (x) : (y))

#define S1(i,j) { hash(1); hash(i); hash(j); }
#define S2(i,j) { hash(2); hash(i); hash(j); }

void test(int M, int N)
{
  /* Original iterators. */
  int i, j;
  for (i=1;i<=2;i++) {
    for (j=1;j<=M;j++) {
      S1(i,j) ;
    }
  }
  for (i=3;i<=M-1;i++) {
    for (j=1;j<=i-1;j++) {
      S1(i,j) ;
    }
    S1(i,i) ;
    S2(i,i) ;
    for (j=i+1;j<=M;j++) {
      S1(i,j) ;
    }
  }
  for (j=1;j<=M-1;j++) {
    S1(M,j) ;
  }
  S1(M,M) ;
  S2(M,M) ;
  for (i=M+1;i<=N;i++) {
    for (j=1;j<=M;j++) {
      S1(i,j) ;
    }
    S2(i,i) ;
  }
}
