/* Generated from ../../../git/cloog/test/lu2.cloog by CLooG 0.14.0-238-gb1cb779 gmp bits in 0.01s. */
if (n >= 2) {
  for (l=2;l<=n;l++) {
    S1(1,n,1,l);
  }
  for (i=2;i<=n-1;i++) {
    for (j=2;j<=n;j++) {
      for (k=1;k<=min(i-1,j-1);k++) {
        S2(i,j,k,j,i);
      }
    }
    for (l=i+1;l<=n;l++) {
      S1(i,n,i,l);
    }
  }
  for (j=2;j<=n;j++) {
    for (k=1;k<=j-1;k++) {
      S2(n,j,k,j,n);
    }
  }
}
