from lark import Lark
import io
import contextlib
from accipit_interpreter import accipit_grammar, accipit_transformer, env, eval
import pytest

def helper_test_with_text(text: str, answer: str):
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="program")
    env.clear()
    parser.parse(text)

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        eval()
        
    assert output.getvalue() == answer

def test_ident():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="ident")
    env.clear()
    parser.parse("@a")
    env.clear()
    parser.parse("%1")
    env.clear()
    parser.parse("#a.b.c")
    
def test_const():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="const")
    env.clear()
    parser.parse("3")
    env.clear()
    parser.parse("none")
    env.clear()
    parser.parse("()")
    env.clear()
    parser.parse("+3")
    env.clear()
    parser.parse("-3")
    
def test_type():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="type")
    env.clear()
    parser.parse("i32")
    env.clear()
    parser.parse("()")
    env.clear()
    parser.parse("i32*")
    env.clear()
    parser.parse("fn(i32, i32**) -> i32")

def test_binexpr():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="bin_expr")
    env.clear()
    parser.parse("add 3, %1")
    env.clear()
    parser.parse("sub @1, %b")
    
def test_terminator():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="terminator")
    env.clear()
    parser.parse("br @a, label %true, label %false")
    env.clear()
    parser.parse("jmp label %b")
    env.clear()
    parser.parse("ret 3")
    
def test_alloca():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="alloca")
    env.clear()
    parser.parse("alloca i32, 3")

def test_load():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="load")
    env.clear()
    parser.parse("load @a")

def test_store():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="store")
    env.clear()
    parser.parse("store 3, %b")

def test_gep():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="gep")
    env.clear()
    parser.parse("offset i32, %a, [2 < 3], [4 < none]")

def test_fncall():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="fncall")
    env.clear()
    parser.parse("call @a, 3, %b")

def test_let():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="value_binding")
    env.clear()
    parser.parse("let @a: i32 = add 3, %b")
    env.clear()
    parser.parse("let %1 = alloca i32, 3")
    env.clear()
    parser.parse("let %2 = load @a")
    env.clear()
    parser.parse("let %3 = store 3, %b")
    env.clear()
    parser.parse("let %4 = offset i32, %a, [2 < 3], [4 < none]")
    env.clear()
    parser.parse("let %5 = call @a, 3, %b")
    
def test_plist():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="plist")
    env.clear()
    parser.parse("#1: i32, #2: i32*")
    env.clear()
    parser.parse("")
    
def test_label():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="label")
    env.clear()
    parser.parse("%L1 :")
    
def test_bb():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="bb")
    env.clear()
    parser.parse("""
        %L1 :
            let %a = add 3, %b
            br @a, label %true, label %false
    """)
    
def test_body():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="body")
    env.clear()
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
    env.clear()
    parser.parse("@a : region i32 ,3")
    env.clear()
    parser.parse("@b : region i32 , 3 = [1, 2, 3]")
    
def test_fun_defn():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="fun_defn")
    env.clear()
    parser.parse("fn @a(#1: i32, #2: i32) -> i32 { %Lentry: ret 3 }")
    
def test_fun_decl():
    parser = Lark(accipit_grammar, parser="lalr", transformer=accipit_transformer, start="fun_decl")
    env.clear()
    parser.parse("fn @a (#1: i32, #2: i32) -> i32 ;")
    
    
def test_eval_1():
    text = """
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
        let %7 = call @write, %6
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

    fn @main() -> i32 {
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
        ret 0
    }
    """
    answer = "5\n6\n3\n4\n1\n2\n7\n8\n"
    helper_test_with_text(text, answer)
    
def test_eval_2():
    text = """
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
        let %3 = call @write, %1
        let %5 = call @write, %2
        ret ()
    }
    """
    answer = "10\n11\n"

def test_eval_3():
    text = """
    fn @factorial(#n: i32) -> i32 {
    %Lentry:
        // create a stack slot of i32 type as the space of the return value.
        // if n equals 1, store `1` to this address, i.e. `return 1`,
        // otherwise, do recursive call, i.e. return n * factorial(n - 1).
        let %ret.addr = alloca i32, 1
        // store function parameter on the stack.
        let %n.addr = alloca i32, 1
        let %4 = store #n, %n.addr
        // create a slot for local variable ans, uninitialized.
        let %ans.addr = alloca i32, 1
        // when we need #n, you just read it from %n.addr.
        let %6 = load %n.addr
        // comparison produce an `i8` value.
        let %cmp = eq %6, 0
        br %cmp, label %Ltrue, label %Lfalse
    %Ltrue:
        // retuen value = 1.
        let %10 = store 1, %ret.addr
        jmp label %Lret
    %Lfalse:
        // n - 1
        let %13 = load %n.addr
        let %14 = sub %13, 1
        // factorial(n - 1)
        let %res = call @factorial, %14
        // n
        let %16 = load %n.addr
        // n * factorial(n - 1)
        let %17 = mul %16, %res
        // write local variable `ans`
        let %18 = store %17, %ans.addr
        // now we meets `return ans`, which means we
        // should first read value from `%ans.addr` and then
        // write it to `%ret.addr`.
        let %19 = load %ans.addr
        let %20 = store %19, %ret.addr
        jmp label %Lret
    %Lret:
        // load return value from %ret.addr
        let %ret.val = load %ret.addr
        ret %ret.val
    }
    fn @main() -> i32 {
    %Lentry:
        let %res = call @factorial, 5
        let %0 = call @write, %res
        ret 0
    }
    """
    answer = "120\n"
    helper_test_with_text(text, answer)