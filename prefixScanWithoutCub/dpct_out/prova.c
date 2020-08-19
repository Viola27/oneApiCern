#include <stdio.h>

int main(void) {
  int size = 256;
  int ci[size];
  int co[size];
  int ws[16];
  int i, riga, offset;
  int warp = 16;

  if (size >= 1024) {
    printf("Errore,troppo grande\n");
    return -1;
  }

  for (i = 0; i < size; i++) {
    ci[i] = i;
    co[i] = 0;
  }

  for (riga = 0; riga < size; riga += warp) {
    // warp prefix scan
    for (offset = 1; offset < 16; offset <<= 1) {
      for (i = riga + warp - 1; i >= riga + offset; i--) {
        ci[i] += ci[i - offset];
      }
    }
    for (i = riga; i < riga + warp; i++) {
      co[i] = ci[i];
    }
    // fine warp prefix scan
    ws[riga / 16] = co[i - 1];
    /*printf("riga: %d | i-1: %d | ws[riga/16]: %d\n", riga / 16, i - 1,
           ws[riga / 16]);*/
  }
  /*
    printf("\nci:");
    for (i = 0; i < size; i++) {
      if (!(i % 16))
        printf("\n");
      printf(" %d ", ci[i]);
    }
    printf("\nco:");
    for (i = 0; i < size; i++) {
      if (!(i % 16))
        printf("\n");
      printf(" %d ", co[i]);
    }
  */
  if (size <= 16)
    return 0;

  for (i = 0; i < warp; i++) { // prima riga
    // warp prefix scan
    for (offset = 1; offset < 16; offset <<= 1) {
      for (i = offset; i < warp; i++) {
        ws[i] += ws[i - offset];
      }
    }
    // fine warp prefix scan
  }

  for (i = 16; i < size; i++) { // per tutte le righe tranne la prima
    riga = i / 16;
    co[i] += ws[riga - 1];
  }

  co[0] = ci[0];
  for (i = 1; i < size; ++i)
    co[i] = ci[i] + co[i - 1];

  // stampe
  printf("\nci:");
  for (i = 0; i < size; i++) {
    if (!(i % 16))
      printf("\n");
    printf(" %d ", ci[i]);
  }
  printf("\nco:");
  for (i = 0; i < size; i++) {
    if (!(i % 16))
      printf("\n");
    printf(" %d ", co[i]);
  }

  printf("\nws:");
  for (i = 0; i < 16; i++) {
    if (!(i % 16))
      printf("\n");
    printf(" %d", ws[i]);
  }
  return 0;
}