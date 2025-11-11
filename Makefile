TOPTARGETS := clean

SUBDIRS := $(shell find -name "*src")
SUBDIRS2 := $(shell find -name "*src[0-9]*")

$(TOPTARGETS): $(SUBDIRS)
$(SUBDIRS):
	echo "make arg is" $(MAKECMDGOALS)
	$(MAKE) -C $@ $(MAKECMDGOALS)

SUBCLEAN = $(addsuffix .clean,$(SUBDIRS))
SUBCLEAN2 = $(addsuffix .clean2,$(SUBDIRS2))

clean: $(SUBCLEAN) $(SUBCLEAN2)

$(SUBCLEAN): %.clean:
	$(MAKE) -C $* clean

$(SUBCLEAN2): %.clean2:
	rm -rf $*
