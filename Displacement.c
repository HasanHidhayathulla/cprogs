#include<stdio.h>
#include<math.h>
float displac(int x,float a[x][2]);
int main(){
    int n;
    printf("Enter the amount of diversions");
    scanf("%d",&n);
    float journey[n][2];
    printf("\ntip : East-0, South-1, West-2, North-3");
    for (int i = 0; i < n; i++){
        printf("\nDirection, distance");
        scanf("%f %f", &journey[i][0], &journey[i][1]);
    }
    float displ= displac(n,journey);
    printf("\nNet displacement = %.2f units\n", displ);
    return 0;

}

float displac(int x,float a[x][2]){
    float vert = 0, horz = 0;
    for (int i = 0; i < x; i++){
        if (a[i][0]==0)horz = horz + a[i][1];
        if (a[i][0]==1)vert = vert - a[i][1];
        if (a[i][0]==2)horz = horz - a[i][1];
        if (a[i][0]==3)vert = vert + a[i][1];
    }
    //if (horz == 0 | vert == 0)return horz+vert;
    return sqrt((vert*vert)+(horz*horz));
}
