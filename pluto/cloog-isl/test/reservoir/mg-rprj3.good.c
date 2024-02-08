/* Generated from ../../../git/cloog/test/./reservoir/mg-rprj3.cloog by CLooG 0.14.0-72-gefe2fc2 gmp bits in 0.39s. */
extern void hash(int);

/* Useful macros. */
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define max(x,y)    ((x) > (y) ? (x) : (y))
#define min(x,y)    ((x) < (y) ? (x) : (y))

#define S1(i,j,k) { hash(1); hash(i); hash(j); hash(k); }
#define S2(i,j,k) { hash(2); hash(i); hash(j); hash(k); }
#define S3(i,j,k) { hash(3); hash(i); hash(j); hash(k); }
#define S4(i,j,k) { hash(4); hash(i); hash(j); hash(k); }
#define S5(i,j,k) { hash(5); hash(i); hash(j); hash(k); }

void test(int M, int N, int O, int P, int Q, int R)
{
  /* Scattering iterators. */
  int c2, c4, c6;
  /* Original iterators. */
  int i, j, k;
  if ((M >= 4) && (N >= 4)) {
    for (c2=2;c2<=O-1;c2++) {
      for (c6=2;c6<=M;c6++) {
        S1(c2,2,c6) ;
      }
      for (c4=3;c4<=N-1;c4++) {
        for (c6=2;c6<=M;c6++) {
          j = c4-1 ;
          S2(c2,c4-1,c6) ;
        }
        j = c4-1 ;
        S4(c2,c4-1,2) ;
        for (c6=2;c6<=M-2;c6++) {
          j = c4-1 ;
          S3(c2,c4-1,c6) ;
          j = c4-1 ;
          S5(c2,c4-1,c6) ;
          j = c4-1 ;
          k = c6+1 ;
          S4(c2,c4-1,c6+1) ;
        }
        c6 = M-1 ;
        j = c4-1 ;
        k = M-1 ;
        S3(c2,c4-1,M-1) ;
        j = c4-1 ;
        k = M-1 ;
        S5(c2,c4-1,M-1) ;
        for (c6=2;c6<=M;c6++) {
          S1(c2,c4,c6) ;
        }
      }
      for (c6=2;c6<=M;c6++) {
        j = N-1 ;
        S2(c2,N-1,c6) ;
      }
      j = N-1 ;
      S4(c2,N-1,2) ;
      for (c6=2;c6<=M-2;c6++) {
        j = N-1 ;
        S3(c2,N-1,c6) ;
        j = N-1 ;
        S5(c2,N-1,c6) ;
        j = N-1 ;
        k = c6+1 ;
        S4(c2,N-1,c6+1) ;
      }
      c6 = M-1 ;
      j = N-1 ;
      k = M-1 ;
      S3(c2,N-1,M-1) ;
      j = N-1 ;
      k = M-1 ;
      S5(c2,N-1,M-1) ;
    }
  }
  if ((M >= 4) && (N == 3)) {
    for (c2=2;c2<=O-1;c2++) {
      for (c6=2;c6<=M;c6++) {
        S1(c2,2,c6) ;
      }
      for (c6=2;c6<=M;c6++) {
        S2(c2,2,c6) ;
      }
      S4(c2,2,2) ;
      for (c6=2;c6<=M-2;c6++) {
        S3(c2,2,c6) ;
        S5(c2,2,c6) ;
        k = c6+1 ;
        S4(c2,2,c6+1) ;
      }
      c6 = M-1 ;
      k = M-1 ;
      S3(c2,2,M-1) ;
      k = M-1 ;
      S5(c2,2,M-1) ;
    }
  }
  if ((M == 3) && (N == 3)) {
    for (c2=2;c2<=O-1;c2++) {
      for (c6=2;c6<=3;c6++) {
        S1(c2,2,c6) ;
      }
      for (c6=2;c6<=3;c6++) {
        S2(c2,2,c6) ;
      }
      S4(c2,2,2) ;
      S3(c2,2,2) ;
      S5(c2,2,2) ;
    }
  }
  if ((M == 3) && (N >= 4)) {
    for (c2=2;c2<=O-1;c2++) {
      for (c6=2;c6<=3;c6++) {
        S1(c2,2,c6) ;
      }
      for (c4=3;c4<=N-1;c4++) {
        for (c6=2;c6<=3;c6++) {
          j = c4-1 ;
          S2(c2,c4-1,c6) ;
        }
        j = c4-1 ;
        S4(c2,c4-1,2) ;
        j = c4-1 ;
        S3(c2,c4-1,2) ;
        j = c4-1 ;
        S5(c2,c4-1,2) ;
        for (c6=2;c6<=3;c6++) {
          S1(c2,c4,c6) ;
        }
      }
      for (c6=2;c6<=3;c6++) {
        j = N-1 ;
        S2(c2,N-1,c6) ;
      }
      j = N-1 ;
      S4(c2,N-1,2) ;
      j = N-1 ;
      S3(c2,N-1,2) ;
      j = N-1 ;
      S5(c2,N-1,2) ;
    }
  }
  if ((M == 2) && (N >= 4)) {
    for (c2=2;c2<=O-1;c2++) {
      S1(c2,2,2) ;
      for (c4=3;c4<=N-1;c4++) {
        j = c4-1 ;
        S2(c2,c4-1,2) ;
        S1(c2,c4,2) ;
      }
      j = N-1 ;
      S2(c2,N-1,2) ;
    }
  }
  if ((M == 2) && (N == 3)) {
    for (c2=2;c2<=O-1;c2++) {
      S1(c2,2,2) ;
      S2(c2,2,2) ;
    }
  }
}
