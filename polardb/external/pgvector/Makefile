EXTENSION = vector
EXTVERSION = 0.8.1

MODULE_big = vector
DATA = $(wildcard sql/*--*--*.sql)
DATA_built = sql/$(EXTENSION)--$(EXTVERSION).sql
OBJS = src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o
# ==========================================
# 🚀 Mimalloc 强制集成 (核弹模式)
# ==========================================

# 1. 强制将头文件路径加入 CPPFLAGS (预处理器参数)
#    override 关键字确保即使系统想忽略也能强制生效
#    $(CURDIR) 确保使用的是绝对路径
override CPPFLAGS += -I$(CURDIR)/mimalloc/include -DMI_MALLOC_OVERRIDE

# 2. 【关键修正】禁用严格检查
#    -Wno-missing-prototypes: 允许函数没有声明 (解决 alloc-override.c 的报错)
#    -Wno-error: 防止其他警告变成错误导致编译失败
override PG_CFLAGS += -Wno-missing-prototypes -Wno-error

# 2. 将源码加入编译列表
OBJS += mimalloc/src/static.o

# ==========================================
HEADERS = src/halfvec.h src/sparsevec.h src/vector.h

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test --load-extension=$(EXTENSION)

# To compile for portability, run: make OPTFLAGS=""
OPTFLAGS = -march=native -mtune=native -mavx512f -mavx512dq -mavx512vl -mavx512bw -funroll-loops

# Mac ARM doesn't always support -march=native
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		# no difference with -march=armv8.5-a
		OPTFLAGS =
	endif
endif

# PowerPC doesn't support -march=native
ifneq ($(filter ppc64%, $(shell uname -m)), )
	OPTFLAGS =
endif

# For auto-vectorization:
# - GCC (needs -ftree-vectorize OR -O3) - https://gcc.gnu.org/projects/tree-ssa/vectorization.html
# - Clang (could use pragma instead) - https://llvm.org/docs/Vectorizers.html
PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math
# # ==========================================
# # 🚀 Mimalloc 集成配置 (绝对路径修正版)
# # ==========================================

# # 1. 使用 $(srcdir) 锁定源码目录的绝对位置
# #    这是解决 "No such file" 的关键！
# MIMALLOC_PATH = $(srcdir)/mimalloc

# # 2. 添加头文件路径 (使用绝对路径)
# PKG_CPPFLAGS += -I$(MIMALLOC_PATH)/include -DMI_MALLOC_OVERRIDE

# # 3. 将源码加入编译列表
# #    注意：这里依然保持相对路径，因为 make 会根据目录结构自动匹配 .c 文件
# OBJS += mimalloc/src/static.o

# # ==========================================

# Debug GCC auto-vectorization
# PG_CFLAGS += -fopt-info-vec

# Debug Clang auto-vectorization
# PG_CFLAGS += -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize

all: sql/$(EXTENSION)--$(EXTVERSION).sql

sql/$(EXTENSION)--$(EXTVERSION).sql: sql/$(EXTENSION).sql
	cp $< $@

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = external/pgvector
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif

# for Mac
ifeq ($(PROVE),)
	PROVE = prove
endif

# for Postgres < 15
PROVE_FLAGS += -I ./test/perl

prove_installcheck:
	rm -rf $(CURDIR)/tmp_check
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='6$(DEF_PGPORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) $(if $(PROVE_TESTS),$(PROVE_TESTS),test/t/*.pl)

.PHONY: dist

dist:
	mkdir -p dist
	git archive --format zip --prefix=$(EXTENSION)-$(EXTVERSION)/ --output dist/$(EXTENSION)-$(EXTVERSION).zip master

# for Docker
PG_MAJOR ?= 17

.PHONY: docker

docker:
	docker build --pull --no-cache --build-arg PG_MAJOR=$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) .

.PHONY: docker-release

docker-release:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) --build-arg DEBIAN_CODENAME=bookworm -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR)-bookworm -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR)-bookworm .

.PHONY: docker-release-trixie

docker-release-trixie:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) --build-arg DEBIAN_CODENAME=trixie -t pgvector/pgvector:pg$(PG_MAJOR)-trixie -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR)-trixie .
