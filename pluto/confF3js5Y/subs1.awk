BEGIN {
S["am__EXEEXT_FALSE"]=""
S["am__EXEEXT_TRUE"]="#"
S["LTLIBOBJS"]=""
S["BASH"]="/bin/bash"
S["EXTERNAL_ISL_FALSE"]=""
S["EXTERNAL_ISL_TRUE"]="#"
S["ISL_LIBADD"]=""
S["ISL_LDFLAGS"]=""
S["ISL_INCLUDE"]=""
S["LIBOBJS"]=""
S["FILECHECK"]="yes"
S["EXTERNAL_CLANG_FALSE"]=""
S["EXTERNAL_CLANG_TRUE"]="#"
S["BUILD_DIR"]="/home/huangl/workspace/RfCgraTrans/pluto"
S["GUIDING_DECLS"]=""
S["PLUTO_DEBUG_FALSE"]=""
S["PLUTO_DEBUG_TRUE"]="#"
S["CXXCPP"]="/usr/bin/c++ -std=c++11 -E"
S["CPP"]="/usr/bin/cc -E"
S["LT_SYS_LIBRARY_PATH"]=""
S["OTOOL64"]=""
S["OTOOL"]=""
S["LIPO"]=""
S["NMEDIT"]=""
S["DSYMUTIL"]=""
S["MANIFEST_TOOL"]=":"
S["RANLIB"]="ranlib"
S["ac_ct_AR"]="ar"
S["AR"]="ar"
S["DLLTOOL"]="false"
S["OBJDUMP"]="objdump"
S["LN_S"]="ln -s"
S["NM"]="/usr/bin/nm -B"
S["ac_ct_DUMPBIN"]=""
S["DUMPBIN"]=""
S["LD"]="/usr/bin/ld -m elf_x86_64"
S["FGREP"]="/usr/bin/grep -F"
S["EGREP"]="/usr/bin/grep -E"
S["GREP"]="/usr/bin/grep"
S["SED"]="/usr/bin/sed"
S["host_os"]="linux-gnu"
S["host_vendor"]="pc"
S["host_cpu"]="x86_64"
S["host"]="x86_64-pc-linux-gnu"
S["build_os"]="linux-gnu"
S["build_vendor"]="pc"
S["build_cpu"]="x86_64"
S["build"]="x86_64-pc-linux-gnu"
S["LIBTOOL"]="$(SHELL) $(top_builddir)/libtool"
S["am__fastdepCC_FALSE"]="#"
S["am__fastdepCC_TRUE"]=""
S["CCDEPMODE"]="depmode=gcc3"
S["ac_ct_CC"]="/usr/bin/cc"
S["CFLAGS"]="-Wall"
S["CC"]="/usr/bin/cc"
S["HAVE_CXX11"]="1"
S["am__fastdepCXX_FALSE"]="#"
S["am__fastdepCXX_TRUE"]=""
S["CXXDEPMODE"]="depmode=gcc3"
S["am__nodep"]="_no"
S["AMDEPBACKSLASH"]="\\"
S["AMDEP_FALSE"]="#"
S["AMDEP_TRUE"]=""
S["am__include"]="include"
S["DEPDIR"]=".deps"
S["OBJEXT"]="o"
S["EXEEXT"]=""
S["ac_ct_CXX"]=""
S["CPPFLAGS"]=""
S["LDFLAGS"]=""
S["CXXFLAGS"]="-Wall -std=c++11"
S["CXX"]="/usr/bin/c++ -std=c++11"
S["AM_BACKSLASH"]="\\"
S["AM_DEFAULT_VERBOSITY"]="0"
S["AM_DEFAULT_V"]="$(AM_DEFAULT_VERBOSITY)"
S["AM_V"]="$(V)"
S["am__untar"]="$${TAR-tar} xf -"
S["am__tar"]="$${TAR-tar} chof - \"$$tardir\""
S["AMTAR"]="$${TAR-tar}"
S["am__leading_dot"]="."
S["SET_MAKE"]=""
S["AWK"]="gawk"
S["mkdir_p"]="$(MKDIR_P)"
S["MKDIR_P"]="/usr/bin/mkdir -p"
S["INSTALL_STRIP_PROGRAM"]="$(install_sh) -c -s"
S["STRIP"]="strip"
S["install_sh"]="${SHELL} /home/huangl/workspace/RfCgraTrans/pluto/install-sh"
S["MAKEINFO"]="${SHELL} /home/huangl/workspace/RfCgraTrans/pluto/missing makeinfo"
S["AUTOHEADER"]="${SHELL} /home/huangl/workspace/RfCgraTrans/pluto/missing autoheader"
S["AUTOMAKE"]="${SHELL} /home/huangl/workspace/RfCgraTrans/pluto/missing automake-1.16"
S["AUTOCONF"]="${SHELL} /home/huangl/workspace/RfCgraTrans/pluto/missing autoconf"
S["ACLOCAL"]="${SHELL} /home/huangl/workspace/RfCgraTrans/pluto/missing aclocal-1.16"
S["VERSION"]="0.12.0"
S["PACKAGE"]="pluto"
S["CYGPATH_W"]="echo"
S["am__isrc"]=""
S["INSTALL_DATA"]="${INSTALL} -m 644"
S["INSTALL_SCRIPT"]="${INSTALL}"
S["INSTALL_PROGRAM"]="${INSTALL}"
S["target_alias"]=""
S["host_alias"]=""
S["build_alias"]=""
S["LIBS"]=""
S["ECHO_T"]=""
S["ECHO_N"]="-n"
S["ECHO_C"]=""
S["DEFS"]="-DHAVE_CONFIG_H"
S["mandir"]="${datarootdir}/man"
S["localedir"]="${datarootdir}/locale"
S["libdir"]="${exec_prefix}/lib"
S["psdir"]="${docdir}"
S["pdfdir"]="${docdir}"
S["dvidir"]="${docdir}"
S["htmldir"]="${docdir}"
S["infodir"]="${datarootdir}/info"
S["docdir"]="${datarootdir}/doc/${PACKAGE_TARNAME}"
S["oldincludedir"]="/usr/include"
S["includedir"]="${prefix}/include"
S["runstatedir"]="${localstatedir}/run"
S["localstatedir"]="${prefix}/var"
S["sharedstatedir"]="${prefix}/com"
S["sysconfdir"]="${prefix}/etc"
S["datadir"]="${datarootdir}"
S["datarootdir"]="${prefix}/share"
S["libexecdir"]="${exec_prefix}/libexec"
S["sbindir"]="${exec_prefix}/sbin"
S["bindir"]="${exec_prefix}/bin"
S["program_transform_name"]="s,x,x,"
S["prefix"]="/home/huangl/workspace/RfCgraTrans/build/pluto"
S["exec_prefix"]="${prefix}"
S["PACKAGE_URL"]=""
S["PACKAGE_BUGREPORT"]="udayb@iisc.ac.in"
S["PACKAGE_STRING"]="pluto 0.12.0"
S["PACKAGE_VERSION"]="0.12.0"
S["PACKAGE_TARNAME"]="pluto"
S["PACKAGE_NAME"]="pluto"
S["PATH_SEPARATOR"]=":"
S["SHELL"]="/bin/bash"
S["am__quote"]=""
  for (key in S) S_is_set[key] = 1
  FS = ""

}
{
  line = $ 0
  nfields = split(line, field, "@")
  substed = 0
  len = length(field[1])
  for (i = 2; i < nfields; i++) {
    key = field[i]
    keylen = length(key)
    if (S_is_set[key]) {
      value = S[key]
      line = substr(line, 1, len) "" value "" substr(line, len + keylen + 3)
      len += length(value) + length(field[++i])
      substed = 1
    } else
      len += 1 + keylen
  }

  print line
}
