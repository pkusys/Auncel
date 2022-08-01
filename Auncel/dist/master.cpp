// Copyright (c) Zili Zhang.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<unistd.h>
#include<sys/time.h>

#include <sys/wait.h>


/// CPP part
#include<string>
#include<iostream>
#include<vector>

#define MAXLINE 4096

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

bool call(std::string ip, std::string ins, int index = -1){
    int sockfd, n;
    char recvline[4096], sendline[4096];
    struct sockaddr_in  servaddr;

    if( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        printf("create socket error: %s(errno: %d)\n", strerror(errno),errno);
        return 0;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(3456);
    if( inet_pton(AF_INET, ip.c_str(), &servaddr.sin_addr) <= 0){
        printf("inet_pton error for %s\n", ip.c_str());
        return 0;
    }

    if( connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0){
        printf("connect error: %s(errno: %d)\n",strerror(errno),errno);
        return 0;
    }

    printf("send msg to server: \n");
    printf("sendline: %s\n", ins.c_str());
    if( send(sockfd, ins.c_str(), strlen(ins.c_str()), 0) < 0){
        printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
        return 0;
    }
    if(ins == "search"){
        // recv files
        char buffer[MAXLINE];
        std::string file_end = "file done";
        std::string file_idx ="idx.ivecs", file_dis = "dis.fvecs";
        file_idx = std::to_string(index) + file_idx;
        file_dis = std::to_string(index) + file_dis;
        FILE *fp, *fp2;
        fp = fopen(file_idx.c_str(), "wb");
        while(1){
            n = recv(sockfd, buffer, MAXLINE, 0);
            if(n == 0)
                break;
            if(n == strlen(file_end.c_str()) && buffer[0] == 'f' && buffer[5] == 'd'){
                break;
            }
            fwrite(buffer, 1, n, fp);
        }
        fp2 = fopen(file_dis.c_str(), "wb");
        while(1){
            n = recv(sockfd, buffer, MAXLINE, 0);
            if(n == 0)
                break;
            if(n == strlen(file_end.c_str()) && buffer[0] == 'f' && buffer[5] == 'd')
                break;
            fwrite(buffer, 1, n, fp2);
        }
        fclose(fp);
        fclose(fp2);
    }
    int tmp = recv(sockfd, recvline, MAXLINE, 0);
    recvline[tmp] = '\0';
    printf("recv msg from sever: %s\n", recvline);
    bool res = false;
    std::string ret = recvline;
    if (ret == "success")
        res = true;
    close(sockfd);
    return res;
}

void sys_error(const char* str)
{
    perror(str);
    exit(1);
}

int main(int argc, char** argv){
    pid_t pid;
    int i = 0;
    std::vector<std::string> ips = {"162.105.19.157", "162.105.19.238"};
    for(;i < ips.size(); i++){
        if((pid = fork())==0)
            break;
        if(pid == -1)
            sys_error("fork error");
    }
    // Parent process
    if(pid > 0){
        int status,wpid;
        while ((wpid=waitpid(-1,&status,0))!= -1)   // blocked here
        {
             printf("wpid = %d\n",wpid);
        }
        printf("All done\n");
        double t0 = elapsed();
        i = 0;
        for(;i < ips.size(); i++){
            if((pid = fork())==0)
                break;
            if(pid == -1)
                sys_error("fork error");
        }
        if(pid > 0){
            int status,wpid;
            while ((wpid=waitpid(-1,&status,0))!= -1)   // blocked here
            {
                printf("wpid = %d\n",wpid);
            }
            printf("All search done: %.3lf\n", elapsed() - t0);
        }
        else{
            std::string ip = ips[i];
            if(call(ip, "search", i))
                printf("Search in ip: %s finished\n", ip.c_str());
            else
                printf("Search in ip: %s failed\n", ip.c_str());
        }
    }
    // Child process
    else{
        std::string ip = ips[i];
        if(call(ip, "train"))
            printf("Train in ip: %s finished\n", ip.c_str());
        else
            printf("Train in ip: %s failed\n", ip.c_str());
    }

    exit(0);
}