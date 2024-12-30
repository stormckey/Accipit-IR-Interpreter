from lark import Lark, Transformer, ast_utils
from accipit_interpreter import accipit_grammar, accipit_transformer

def test_ident():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="ident")
    parser.parse("@a")
    parser.parse("%1")
    parser.parse("#a.b.c")
    
def test_const():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="const")
    parser.parse("3")
    parser.parse("none")
    parser.parse("()")
    parser.parse("+3")
    parser.parse("-3")
    
def test_type():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="type")
    parser.parse("i32")
    parser.parse("()")
    parser.parse("i32*")
    parser.parse("fn(i32, i32**) -> i32")

def test_binexpr():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="binexpr")
    parser.parse("add 3, %1")
    parser.parse("sub @1, %b")
    
def test_terminator():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="terminator")
    parser.parse("br @a, label %true, label %false")
    parser.parse("jmp label %b")
    parser.parse("ret 3")
    
def test_alloca():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="alloca")
    parser.parse("alloca i32, 3")

def test_load():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="load")
    parser.parse("load @a")

def test_store():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="store")
    parser.parse("store 3, %b")

def test_gep():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="gep")
    parser.parse("offset i32, %a, [2 < 3], [4 < none]")

def test_fncall():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="fncall")
    parser.parse("call @a, 3, %b")

def test_let():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="valuebinding")
    parser.parse("let @a: i32 = add 3, %b")
    parser.parse("let %a = alloca i32, 3")
    parser.parse("let %a = load @a")
    parser.parse("let %a = store 3, %b")
    parser.parse("let %a = offset i32, %a, [2 < 3], [4 < none]")
    parser.parse("let %a = call @a, 3, %b")
    
def test_plist():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="plist")
    parser.parse("#1: i32, %2: i32*")
    parser.parse("")
    
def test_label():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="label")
    parser.parse("%L1 :")
    
def test_bb():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="bb")
    parser.parse("""
        %L1 :
            let %a = add 3, %b
            br @a, label %true, label %false
    """)
    
def test_body():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="body")
    parser.parse("""
        {
            %Lentry:
                /* Create a stack slot of i32 type as the space of the return value.
                 * if n equals 1, store `1` to this address, i.e. `return 1`,
                 * otherwise, do recursive call, i.e. return n * factorial(n - 1).
                 */
                let %ret.addr = alloca i32, 1
                let %cmp = eq #n, 0
                br %cmp, label %Ltrue, label %Lfalse
            %Ltrue:
                let %6 = store 1, %ret.addr
                jmp label %Lret
            %Lfalse:
                let %9 = sub #n, 1
                let %res = call @factorial, %9
                let %11 = mul %9, %res
                let %12 = store %11, %ret.addr
                jmp label %Lret
            %Lret:
                let %ret.val = load %ret.addr
                ret %ret.val
        }
    """)

def test_global():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="global_decl")
    parser.parse("@a : region i32 ,3")
    parser.parse("@a : region i32 , 3 = [1, 2, 3]")
    
def test_fun_defn():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="fun_defn")
    parser.parse("fn @a(#1: i32, #2: i32) -> i32 { %Lentry: ret 3 }")
    
def test_fun_decl():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="fun_decl")
    parser.parse("fn @a (#1: i32, #2: i32) -> i32 ;")
    
def test_program():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="program")
    parser.parse("""
        fn @read_write_array(#array.addr: i32*) -> i32 {
        %entry:
            let %size = call @getarray, #array.addr
            let %0 = call @putarray, %size, #array.addr
            ret %size
        }

        fn @main() -> () {
        %0:
            let %array.addr = alloca i32, 10
            let %size = call @read_write_array, %array.addr
            let %1 = call @putch, 10
            let %2 = call @putint, %size
            let %3 = call @putch, 10
            let %index = call @getint
            let %offset = offset i32, %array.addr, [%index < 10]
            let %value = load %offset
            let %4 = call @putint, %value
            ret ()
        }
    """)
    parser.parse("""
        fn @add(#1: i32, #2: i32) -> i32 {
        %3:
            let %4: i32 = add #1, #2
            ret %4
        }

        fn @add_plus_one(#1: i32, #2: i32) -> i32 {
        %3:
            let %4: i32 = add #1, #2
            let %5: i32 = add #4, 1
            ret %5
        }

        fn @add_ret_unit(#1: i32, #2: i32) -> () {
        %3:
            let %4: i32 = add #1, #2
            let %5: i32 = add #4, 1
            ret ()
        }

        fn @add_but_unused_bb(#a: i32, #b: i32) -> i32 {
        %3:
            let %4 = add #a, #b
            ret %4
        %dead_bb:
            let %5: i32 = add #4, 1
            ret %5
        }

        fn @add_but_direct_link_bb(#a: i32, #b: i32) -> i32 {
        %3:
            let %4 = add #a, #b
            jmp label %6
        %dead_bb:
            let %5: i32 = add #4, 1
            ret %5
        %6:
            ret %4
        }

        fn @add_with_load_store_alloca(#1: i32, #2: i32) -> i32 {
        %entry:
            let %arg.1.addr = alloca i32, 1
            let %arg.2.addr = alloca i32, 1
            let %3 = store #1, %arg.1.addr
            let %4 = store #2, %arg.2.addr
            jmp label %6
        %6:
            let %7 = load %arg.1.addr
            let %8 = load %arg.2.addr
            let %9: i32 = add %7, %8
            ret %9
        }

        fn @load_store_alloca_offset(#1: i32, #2: i32) -> i32 {
        %entry:
            let %arg.array = alloca i32, 6
            let %arg.1.addr = offset i32, %arg.array, [0 < 2], [1 < 3]
            let %arg.2.addr = offset i32, %arg.array, [1 < 2], [2 < 3]
            let %3 = store #1, %arg.1.addr
            let %4 = store #2, %arg.2.addr
            jmp label %6
        %6:
            let %8 = load %arg.2.addr
            ret %8
        }
    """)
    parser.parse("""
        // void print_array(int a[], int len)
        // This is an optimized manually written version.
        // NOTE: The output heavily relies on the assumption that
        // multi-dimension array has a continuous memory, which is not specified in
        // our SysY standard, but rather target dependent.

        fn @print_array(#a: i32*, #len: i32) -> () {
        %entry:
            let %i.addr = alloca i32, 1
            let %0 = store 0, %i.addr
            jmp label %while_entry
        %while_entry:
            let %1 = load %i.addr
            let %4 = lt %1, #len
            br %4, label %while_body, label %while_exit
        %while_body:
            let %i.load = load %i.addr
            let %5 = offset i32, #a, [%i.load < none]
            let %6 = load %5
            let %7 = call @putint, %6
            let %8 = call @putch, 10
            // i = i + 1
            let %9 = load %i.addr
            let %10 = add %9, 1
            let %11 = store %10, %i.addr
            jmp label %while_entry
        %while_exit:
            jmp label %return
        %return:
            ret ()
        }

        fn @main() -> () {
        %entry:
            // int a[4][2]
            let %a.addr = alloca i32, 8
            let %0 = offset i32, %a.addr, [0 < 4], [0 < 2]   
            let %1 = store 1, %0
            let %2 = offset i32, %a.addr, [0 < 4], [1 < 2]   
            let %3 = store 2, %2
            let %4 = offset i32, %a.addr, [1 < 4], [0 < 2]   
            let %5 = store 3, %4
            let %6 = offset i32, %a.addr, [1 < 4], [1 < 2]   
            let %7 = store 4, %6
            let %8 = offset i32, %a.addr, [2 < 4], [0 < 2]   
            let %9 = store 5, %8
            let %10 = offset i32, %a.addr, [2 < 4], [1 < 2]   
            let %11 = store 6, %10
            let %12 = offset i32, %a.addr, [3 < 4], [0 < 2]   
            let %13 = store 7, %12
            let %14 = offset i32, %a.addr, [3 < 4], [1 < 2]   
            let %15 = store 8, %14
            // [2 < 4], [0 < 2] can also be [4 < 8] if you like
            // assuming the target properties mentioned above.
            let %16 = offset i32, %a.addr, [2 < 4], [0 < 2]
            let %17 = call @print_array, %16, 2
            let %18 = offset i32, %a.addr, [1 < 4], [0 < 2]
            let %19 = call @print_array, %18, 2
            let %20 = offset i32, %a.addr, [0 < 4], [0 < 2]
            let %21 = call @print_array, %20, 2
            let %22 = offset i32, %a.addr, [3 < 4], [0 < 2]
            let %23 = call @print_array, %22, 2
            ret ()
        }
    """)
    parser.parse("""
        // a is a global array with 105 i32 elements.
        // Suppose in SysY it is a multi-dimension array `int a[5][3][7]`
        @a : region i32, 105


        fn @write_global_var(#value: i32) -> () {
        %entry:
            // offset: 3 * 3 * 7 + 2 * 7 + 4 = 81
            let %0 = offset i32, @a, [3 < 5], [2 < 3], [4 < 7]
            let %1 = store #value, %0
            // offset based on previous offsets.
            // offset: 81 + 1 * 3 * 7 + 2 = 104
            let %2 = offset i32, %0, [1 < 5], [0 < 3], [2 < 7]
            let %3 = add #value, 1
            let %4 = store %3, %2
            ret ()
        }

        fn @read_global_var(#dim1: i32, #dim2: i32, #dim3: i32) -> i32 {
        %entry:
            let %0 = offset i32, @a, [#dim1 < 5], [#dim2 < 3], [#dim3 < 7]
            let %1 = load %0
            ret %1
        }

        fn @main() -> () {
        %entry:
            let %0 = call @write_global_var, 10
            let %1 = call @read_global_var, 3, 2, 4
            let %2 = call @read_global_var, 4, 2, 6
            let %3 = call @putint, %1
            let %4 = call @putch, 10
            let %5 = call @putint, %2
            ret ()
        }
    """)
    parser.parse("""
        /*
        int if_ifElse_() {
          int a;
          a = 5;
          int b;
          b = 10;
          if(a == 5){
            if (b == 10) 
              a = 25;
            else 
              a = a + 15;
          }
          return (a);
        }
        */

        fn @if_ifElse_() -> i32 {
        %entry:
            let %ret.addr = alloca i32, 1
            // int a;
            let %0 = alloca i32, 1
            let %1 = store 5, %0
            // int b;
            let %2 = alloca i32, 1
            let %3 = store 10, %2
            // a == 5
            let %4 = load %0
            let %5 = eq %4, 5
            br %5, label %outer_if_true, label %outer_if_exit
        %outer_if_true:
            // b == 10
            let %6 = load %2
            let %7 = eq %6, 10
            br %7, label %inner_if_true, label %inner_if_false
        %inner_if_true:
            let %8 = store 25, %0
            jmp label %inner_if_exit
        %inner_if_false:
            let %9 = load %0
            let %10 = add %9, 15
            let %11 = store %10, %0
            jmp label %inner_if_exit
        %inner_if_exit:
            jmp label %outer_if_exit
        %outer_if_exit:
            let %12 = load %0
            let %13 = store %12, %ret.addr
            jmp label %ret_bb
        %ret_bb:
            let %14 = load %ret.addr
            ret %14
        }

        /*
        int if_if_Else() {
          int a;
          a = 5;
          int b;
          b = 10;
          if(a == 5){
            if (b == 10) 
              a = 25;
          }
          else 
              a = a + 15;
          return (a);
        }
        */

        fn @if_if_Else() -> i32 {
        %entry:
            let %ret.addr = alloca i32, 1
            // int a;
            let %0 = alloca i32, 1
            let %1 = store 5, %0
            // int b;
            let %2 = alloca i32, 1
            let %3 = store 10, %2
            // a == 5
            let %4 = load %0
            let %5 = eq %4, 5
            br %5, label %outer_if_true, label %outer_if_false
        %outer_if_true:
            // b == 10
            let %6 = load %2
            let %7 = eq %6, 10
            br %7, label %inner_if_true, label %inner_if_exit
        %inner_if_true:
            let %8 = store 25, %0
            jmp label %inner_if_exit
        %inner_if_exit:
            jmp label %outer_if_exit
        %outer_if_false:
            let %9 = load %0
            let %10 = add %9, 15
            let %11 = store %10, %0
            jmp label %outer_if_exit
        %outer_if_exit:
            let %12 = load %0
            let %13 = store %12, %ret.addr
            jmp label %ret_bb
        %ret_bb:
            let %14 = load %ret.addr
            ret %14
        }

        fn @main() -> () {
        %entry:
            let %0 = call @if_ifElse_
            let %1 = call @putint, %0
            let %2 = call @putch, 10
            let %3 = call @if_if_Else
            let %4 = call @putint, %3
            let %5 = call @putch, 10
            ret ()
        }
    """)