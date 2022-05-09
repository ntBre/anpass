BASE = /home/brent/Projects/rust-anpass
TESTFLAGS = --test-threads=1 --nocapture
SHORT = 0

ifeq (${SHORT}, 0)
TESTFLAGS += --include-ignored
endif

test:
	cargo test -- ${TESTFLAGS} ${ARGS}

# deploy:
# 	RUSTFLAGS='-C target-feature=+crt-static' cargo build --release	\
# 		--target x86_64-unknown-linux-gnu
# 	scp -C ${BASE}/target/x86_64-unknown-linux-gnu/release/intder \
# 		'woods:Programs/brentder/.'

#############
# PROFILING #
#############

profile = RUSTFLAGS='-g' cargo build --release --bin $(1); \
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out	\
		--collect-jumps=yes --simulate-cache=yes		\
		${BASE}/target/release/$(1)

profile.fit:
	$(call profile,fit)

profile.newton:
	$(call profile,newton)
