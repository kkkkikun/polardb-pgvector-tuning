EXTENSION = vector
EXTVERSION = 0.8.1

MODULE_big = vector
DATA = $(wildcard sql/*--*--*.sql)
DATA_built = sql/$(EXTENSION)--$(EXTVERSION).sql
OBJS = src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o
HEADERS = src/halfvec.h src/sparsevec.h src/vector.h

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test --load-extension=$(EXTENSION)

# =========================================================
# 🚀 比赛专用编译配置 (支持 PGO + AVX512 + LTO)
# =========================================================

# 1. 基础极速参数 (Base Flags)
# -march=native: 针对评测机 CPU 架构优化
# -flto: 开启链接时全局优化 (关键!)
# -mavx512...: 强制开启向量指令集
# -march=native -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math
BASE_FLAGS = -march=native -O3 -flto -funroll-loops \
             -mavx512f -mavx512vl -mavx512bw -mavx512dq \
             -mfma -ftree-vectorize -fassociative-math \
             -fno-signed-zeros -fno-trapping-math

# 2. PGO 模式控制 (由 build.sh 传入 PGO=xxx 参数控制)
ifeq ($(PGO),generate)
    # [阶段一] 插桩模式：生成执行统计信息
    PGO_FLAGS = -fprofile-generate
else ifeq ($(PGO),use)
    # [阶段二] 应用模式：读取统计信息进行优化
    # -Wno-missing-profile: 忽略部分未覆盖的代码警告
    # -Wno-coverage-mismatch: 容忍微小的代码行号变化
    PGO_FLAGS = -fprofile-use -Wno-missing-profile -Wno-coverage-mismatch
else
    # 默认模式 (普通编译)
    PGO_FLAGS = 
endif

# 3. 注入编译器参数
# 同时修改 CFLAGS (编译) 和 PG_CFLAGS (扩展编译)
override CFLAGS += $(BASE_FLAGS) $(PGO_FLAGS)
override PG_CFLAGS += $(BASE_FLAGS) $(PGO_FLAGS)

# =========================================================
# override BITCODE_CFLAGS += $(MY_COMPETITION_FLAGS)
# override CFLAGS += -march=native -O3 -funroll-loops -mavx512f -mavx512vl -mavx512bw -mavx512dq

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
# override PG_CFLAGS += -O3 -mfma -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math

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
