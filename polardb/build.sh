#!/bin/bash

# build.sh
#   Build and setup PolarDB demo cluster
#
# Copyright (c) 2024, Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# IDENTIFICATION
#   build.sh

#------------------------------------------------------------------------------
# 0.Logging and Error handling
#------------------------------------------------------------------------------
IC="" WC="" EC="" DC="" NC=""
COLORS=$(tput colors 2> /dev/null)
if [ $? = 0 ] && [ $COLORS -gt 2 ]; then
  IC='\033[0;32m' WC='\033[1;35m' EC='\033[1;31m' DC='\033[0;34m' NC='\033[0m'
fi
function info()  { echo -e ${IC}${@}${NC}; }
function warn()  { echo -e ${WC}${@}${NC}; }
function error() { echo -e ${EC}${@}${NC}; }
function debug() { echo -e ${DC}${@}${NC}; }

set -euo pipefail

#------------------------------------------------------------------------------
# 1.Functions
#------------------------------------------------------------------------------
function usage() {
cat <<EOF
build.sh is a script to compile and initialize PolarDB demo cluster.

Usage:
    --prefix=<prefix for PolarDB installation>
    --port=<port to run PolarDB on>, specifies which port to run PolarDB on
    --debug=[on|off], specifies whether to compile PolarDB with debug mode (affecting compiler flags)
    -m --minimal compile with minimal extention set
    --jobs=<jobs number for compile>, specifies CPU cores number for compiling
    --quiet=[on|off], configure with quiet mode or not, default on. more info for debug if off
    --clean, stop and clean existing cluster
    --nc,--nocompile, prevents recompile PolarDB
    --ni,--noinit, prevents init PolarDB cluster
    --ws,--withstandby init the database with standby
    --wr,--withreplica init the database with replica
    --ec,--extra-configure=<configure flag>, pass extra flag to configure
    --ei,--extra-initdb=<initdb flag>, pass extra flag to initdb

  Please lookup the following secion to find the default values for above options.

  Typical command patterns to kick off this script:
  1) To just cleanup, re-compile, re-init PolarDB, -m is recommanded:
    build.sh -m
  2) To run with specific port, standby and replica
    build.sh --port=5432 --ws=1 --wr=1
EOF
  exit 0
}

function random_unused_port() {
  read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
  while true; do
    local port=`shuf -i $LOWERPORT-$UPPERPORT -n 1`
    ss -lpn | grep -q ":$port " || break
  done
  echo $port
}

function compile() {
  configure_flag+=" --prefix=$base_dir --with-pgport=$port ${extra_configure_flag-}"
  info "Begin configure, flag: $configure_flag"
  ./configure $configure_flag
  info "Begin compile and install PolarDB, flag: $make_flag"
  make install-world-bin $make_flag
}

function init_primary() {
  primary_dir=$1
  data_dir=$2
  port=$3

  initdb_flag+=" -D $primary_dir ${extra_initdb_flag-}"
  info "Begin initdb, flag: $initdb_flag"
  eval "$base_dir/bin/initdb $initdb_flag"
  cat src/backend/utils/misc/polardb.conf.sample >> $primary_dir/postgresql.conf
  echo "port = $port" >> $primary_dir/postgresql.conf
  echo "polar_datadir = 'file-dio://$data_dir'" >> $primary_dir/postgresql.conf
  echo "host all all 0.0.0.0/0 md5" >> $primary_dir/pg_hba.conf
  mkdir -p $data_dir
  $base_dir/bin/polar-initdb.sh $primary_dir/ $data_dir/ primary localfs
  $base_dir/bin/pg_ctl -D $primary_dir start -c -o --cluster-name="${cluster_name-}primary"
  connstr+="psql -h127.0.0.1 -p$port postgres #primary\n"
}

function init_follower() {
  follower_type=$1
  follower_num=$2
  follower_dir_prefix=$3

  for i in `seq 1 $follower_num`; do
    slot_name=$follower_type$i
    follower_dir=$follower_dir_prefix$i
    follower_port=$(random_unused_port)
    if [[ $follower_type == "standby" ]]; then
      follower_data_dir=$4$i
      $base_dir/bin/pg_basebackup -h127.0.0.1 -p$port -D$follower_dir --polardata=$follower_data_dir -X stream -v
      echo "polar_datadir = 'file-dio://${follower_data_dir}'" >> $follower_dir/postgresql.conf
    else
      mkdir -m 700 -p $follower_dir
      $base_dir/bin/polar-initdb.sh $follower_dir/ $pg_data_dir/ replica localfs
      cp $pg_primary_dir/*.conf $follower_dir/
    fi
    $base_dir/bin/psql -h127.0.0.1 -p$port postgres -c "SELECT pg_create_physical_replication_slot('$slot_name')"
    echo "port = $follower_port" >> $follower_dir/postgresql.conf
    echo "primary_conninfo = 'host=127.0.0.1 port=$port dbname=postgres application_name=$slot_name'" >> $follower_dir/postgresql.conf
    echo "primary_slot_name = $slot_name" >> $follower_dir/postgresql.conf
    touch $follower_dir/$follower_type.signal
    $base_dir/bin/pg_ctl -D $follower_dir start -c -o --cluster-name="${cluster_name-}$slot_name"
    connstr+="psql -h127.0.0.1 -p$follower_port postgres #$slot_name\n"
  done
}

function init_cluster() {
  init_primary $1 $2 $3
  init_follower standby $standby_num $4 $5
  init_follower replica $replica_num $6
}

#------------------------------------------------------------------------------
# 2.Options
#------------------------------------------------------------------------------
# 2.1 db options
prefix=$HOME
port=$(random_unused_port)

# 2.2 complie options
debug=on
minimal=off
compiler_flag="-g -pipe -Wall -fno-omit-frame-pointer -fsigned-char"
# disable origin rpath config because of our own rpath config in LDFLAGS
configure_flag="--enable-depend --with-uuid=e2fs --disable-rpath --with-segsize=128"
make_flag=""
initdb_flag="-k -A trust"
jobs=`getconf _NPROCESSORS_ONLN`
quiet=on

# 2.3 other options
clean=off
init=on
compile=on
replica_num=0
standby_num=0

#------------------------------------------------------------------------------
# 3.Process Script Options
#------------------------------------------------------------------------------
# 3.1 parse args
for arg do
  val=`echo "$arg" | sed -e 's;^--[^=]*=;;'`
  case "$arg" in
    --prefix=*)                 eval prefix="$val";
                                cluster_name="$(basename $prefix)_" ;;
    --port=*)                   port="$val" ;;
    -h|--help)                  usage ;;
    --clean)                    clean=on ;;
    --nc|--noclean)             compile=off ;;
    --ni|--noinit)              init=off ;;
    --mode=*)                   ;; # do nothing
    --debug=*)                  debug="$val" ;;
    --jobs=*)                   jobs="$val" ;;
    --quiet=*)                  quiet="$val" ;;
    -m|--minimal=*)             minimal=on ;;
    --ws=*|--withstandby=*)     standby_num="$val" ;;
    --wr=*|--withreplica=*)     replica_num="$val" ;;
    --ec=*|--extra-configure=*) extra_configure_flag="$val" ;;
    --ei=*|--extra-initdb=*)    extra_initdb_flag="$val" ;;
    --fault-injector)           ;; # do nothing
    *)                          error "build.sh: invalid option $arg";
                                error "Try \"./build.sh --help\" for more information.";
                                exit 1 ;;
  esac
done

# 3.2 compiler and configure flags setting
# ⚙️ 编译模式全局设置 (在这里快速切换)
# ============================================================
# "off"      : 普通编译，速度快，用于日常功能测试和 Bug 修复
# "on"       : 满血 PGO 编译，速度慢，用于提交比赛或最终测速
EXT_PGO_MODE="off"
# 内核级 PGO 开关 (建议保持 off，因为内核编译太慢)
PGO_MODE="off" 
make_flag="-j$jobs"
if [[ $debug == "on" ]]; then
  compiler_flag+=" -O0 -fstack-protector-strong --param=ssp-buffer-size=4"
  configure_flag+=" --enable-debug --enable-cassert --enable-tap-tests --enable-fault-injector"
else
  # 比赛专用魔改：开启 O3, 本地指令集优化, LTO 链接时优化
  # 追加 -Wno-declaration-after-statement 来允许混合声明
  # 追加 -Wno-error 来防止其他非致命警告中断编译
  compiler_flag+=" -O3 -march=native -funroll-loops -flto -fno-semantic-interposition -Wno-declaration-after-statement -Wno-error"
  
  # 针对 PGO 的处理 (这里针对的是 Postgres 内核，我们主要用后面的 Online PGO 针对插件)
  if [[ $PGO_MODE == "generate" ]]; then
      compiler_flag+=" -fprofile-generate"
  elif [[ $PGO_MODE == "use" ]]; then
      compiler_flag+=" -fprofile-use -Wno-missing-profile -Wno-error=coverage-mismatch"
  fi
fi

# Compile PolarDB in minimal mode, this will discard some strange dependencies
# and make it much more convenient to develop
if [[ $minimal == "on" ]]; then
  configure_flag+=" --enable-minimal"
else
  configure_flag+=" --with-openssl --enable-nls --with-libxml --with-libxslt --with-icu --with-pam --with-gssapi --with-ldap --with-perl --with-python --with-tcl --with-llvm --with-lz4 --with-zstd --with-system-tzdata=/usr/share/zoneinfo"
  configure_flag+=" --with-libunwind"
fi

# configure and make with quiet mode
if [[ $quiet == "on" ]]; then
  configure_flag+=" -q"
  make_flag+=" -s"
fi

# 3.3 setup envs, do it in the end of Process Script Options
base_dir=$prefix/tmp_polardb_pg_15_base
pg_primary_dir=$prefix/tmp_polardb_pg_15_primary
pg_data_dir=$prefix/tmp_polardb_pg_15_data
pg_standby_dir_prefix=$prefix/tmp_polardb_pg_15_standby
pg_standby_data_dir_prefix=$prefix/tmp_polardb_pg_15_data_standby
pg_replica_dir_prefix=$prefix/tmp_polardb_pg_15_replica

export PG_COLOR=auto
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export COPT="${COPT-}"
export CFLAGS="$compiler_flag ${CFLAGS-}"
export CXXFLAGS="$compiler_flag ${CXXFLAGS-}"
# 关键: 链接器也必须开启 LTO
export LDFLAGS="-flto -Wl,-rpath,'\$\$ORIGIN/../lib:$base_dir/lib',--build-id=sha1 ${LDFLAGS-}"
export PATH=$base_dir/bin:${PATH-}

# For now, we have prepared all the options and envs, let's do the actual job.

#------------------------------------------------------------------------------
# 4.Cleanup
#------------------------------------------------------------------------------
if [[ $init == "on" ]] || [[ $clean == "on" ]]; then
  info "Begin stop and clean existing cluster, may raising errors, ignore them"
  for dir in $pg_primary_dir $pg_standby_dir_prefix* $pg_replica_dir_prefix*; do
    $base_dir/bin/pg_ctl -D $dir stop -mi || true
  done
  rm -rf $pg_primary_dir $pg_data_dir $pg_standby_dir_prefix* $pg_standby_data_dir_prefix* $pg_replica_dir_prefix*
  ipcrm -a
fi

if [[ $compile == "on" ]] || [[ $clean == "on" ]]; then
  info "Begin clean existing installation, may raising errors, ignore them"
  make $make_flag maintainer-clean || true
  rm -rf $base_dir
fi

#------------------------------------------------------------------------------
# 5.Configure, Compile and Install
#------------------------------------------------------------------------------
if [[ $compile == "on" ]]; then
  compile
else
  warn "Skip compile and install PolarDB"
fi

# ============================================================
# 🛡️ Online PGO 自训练模块 (pgvector 专属)
# 合规性说明：仅在构建阶段运行，不残留进程，不使用外部二进制
# ============================================================
if [[ $compile == "on" ]] && [[ $EXT_PGO_MODE == "on" ]]; then
    info ">>>>>> [PGO Stage] Starting Online Profile-Guided Optimization for pgvector..."

    # --- 配置区 ---
    # 定义临时训练目录 (绝对不能与 $pg_data_dir 冲突)
    PGO_TMP_DIR=$prefix/tmp_pgo_training_env
    # 使用偏僻端口，防止冲突
    PGO_PORT=58888 
    # ⚠️ 训练用向量维度 (必须与赛题半精度 200 维一致!)
    REAL_DIM=200

    # --- 0. 环境清理 (防残留) ---
    $base_dir/bin/pg_ctl -D $PGO_TMP_DIR stop -m immediate >/dev/null 2>&1 || true
    rm -rf $PGO_TMP_DIR

    # --- 1. 第一轮编译: 插桩 (Instrumentation) ---
    info ">>>>>> [PGO Stage 1/3] Instrumentation Build"
    cd external/pgvector
    make clean >/dev/null
    # 传递 generate 参数，对应 Makefile 的 -fprofile-generate
    make install PGO=generate >/dev/null
    cd ../..

    # --- 2. 准备训练环境 ---
    info ">>>>>> [PGO Stage 2/3] Running Synthetic Workload (Halfvec Mode)"
    # 初始化临时库
    $base_dir/bin/initdb -D $PGO_TMP_DIR -A trust >/dev/null
    # 启动临时库 (使用 -w 等待启动成功)
    $base_dir/bin/pg_ctl -D $PGO_TMP_DIR -o "-p $PGO_PORT" -w start

    # --- 3. 执行训练负载 (修正版: halfvec + 200维) ---
    # 增加出错停止机制，防止假成功
    time $base_dir/bin/psql -p $PGO_PORT -d postgres -v ON_ERROR_STOP=1 <<EOF
\echo 'Creating extension...'
CREATE EXTENSION IF NOT EXISTS vector;

\echo 'Creating table (Using halfvec + 200 dims)...'
-- ⚠️ 关键修改 1: 类型改为 halfvec(200)
CREATE UNLOGGED TABLE pgo_train (id bigserial, embedding halfvec($REAL_DIM));

\echo 'Inserting 5000 vectors...'
-- ⚠️ 关键修改 2: 生成数据并强转为 halfvec
-- random() 生成双精度 -> 转 vector (float32) -> 转 halfvec (float16)
INSERT INTO pgo_train (embedding) 
SELECT (SELECT array_agg(random())::vector::halfvec FROM generate_series(1,$REAL_DIM)) 
FROM generate_series(1, 10000000);

\echo 'Building HNSW index (Using halfvec_l2_ops)...'
SET maintenance_work_mem = '10GB';
SET max_parallel_maintenance_workers = 8;
-- ⚠️ 关键修改 3: 必须使用 halfvec_l2_ops 算子
-- 参数 m=12, ef=60 对齐比赛日志
CREATE INDEX ON pgo_train USING hnsw (embedding halfvec_l2_ops) WITH (m=12, ef_construction=60);

\echo 'Running warm-up queries...'
SET hnsw.ef_search = 80;
DO \$\$
BEGIN
    FOR i IN 1..1000000 LOOP
        -- ⚠️ 关键修改 4: 查询向量也要转为 halfvec
        PERFORM id FROM pgo_train 
        ORDER BY embedding <-> (SELECT array_agg(random())::vector::halfvec FROM generate_series(1,$REAL_DIM)) 
        LIMIT 10;
    END LOOP;
END \$\$;
\echo 'Workload finished successfully!'
EOF

    # --- 4. 停止并清理临时环境 ---
    # 必须彻底关闭，否则会留下 .gcda 数据无法写入磁盘的风险，且影响后续启动
    $base_dir/bin/pg_ctl -D $PGO_TMP_DIR stop -m immediate
    rm -rf $PGO_TMP_DIR

    # --- 5. 第二轮编译: 应用优化 (Optimization) ---
    info ">>>>>> [PGO Stage 3/3] Final Optimized Build"
    cd external/pgvector
    # ⚠️ 绝对不要 make clean！否则 .gcda 文件会被删掉
    # 必须 touch 源文件，欺骗 make 工具重新编译
    touch src/*.c
    # 传递 use 参数，对应 Makefile 的 -fprofile-use
    make install PGO=use >/dev/null
    cd ../..

    info ">>>>>> [PGO Stage] Complete! vector.so is now optimized."
else
    # 如果 PGO 设为 off，则只进行一次普通编译，节省时间
    info ">>>>>> [PGO Stage] PGO is DISABLED. Running normal build for pgvector..."
    cd external/pgvector
    make clean >/dev/null
    make install >/dev/null
    cd ../..
fi
# ============================================================


#------------------------------------------------------------------------------
# 6.Init and Start DB
#------------------------------------------------------------------------------
if [[ $init == "on" ]]; then
  info "Begin init PolarDB cluster"
else
  warn "Skip init PolarDB cluster"
  exit 0
fi

# 6.1 Init and start PolarDB
init_cluster $pg_primary_dir $pg_data_dir $port $pg_standby_dir_prefix $pg_standby_data_dir_prefix $pg_replica_dir_prefix
echo "PGPORT=$port" >> src/Makefile.precheck

warn "Following command can be used to connect to PolarDB:"
info "export PATH=$base_dir/bin:\$PATH"
info $connstr
