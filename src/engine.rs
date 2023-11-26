use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    None, // Used for root values
    Add,
    Mult,
    Exp,
    Pow,
    Tanh,
}

impl Default for Op {
    fn default() -> Self {
        Op::None
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::None => write!(f, "(none)"),
            Op::Add => write!(f, "+"),
            Op::Mult => write!(f, "*"),
            Op::Exp => write!(f, "exp"),
            Op::Pow => write!(f, "**"),
            Op::Tanh => write!(f, "tanh"),
        }
    }
}

#[derive(Clone)]
enum Input {
    Unary(Scalar),
    Binary(Scalar, Scalar),
}

#[derive(Default)]
struct Node {
    value: f64,
    grad: f64,
    op: Op,
    ins: Option<Input>,
    uuid: Uuid,
}

impl Node {
    fn new(value: f64) -> Self {
        Self {
            value,
            uuid: Uuid::new_v4(),
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub struct Scalar(Rc<RefCell<Node>>);

impl Scalar {
    pub fn new(val: f64) -> Self {
        Self(Rc::new(RefCell::new(Node::new(val))))
    }

    pub fn val(&self) -> f64 {
        self.0.borrow().value
    }

    fn op(&self) -> Op {
        self.0.borrow().op
    }

    fn set_op(&mut self, op: Op) {
        self.0.borrow_mut().op = op;
    }

    fn ins(&self) -> Option<Input> {
        self.0.borrow().ins.clone()
    }

    fn set_ins(&mut self, ins: Input) {
        self.0.borrow_mut().ins = Some(ins);
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    fn set_grad(&mut self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    fn add_grad(&mut self, inc: f64) {
        let cur = self.grad();
        self.set_grad(cur + inc);
    }

    fn uuid(&self) -> Uuid {
        self.0.borrow().uuid
    }

    pub fn tanh(&self) -> Self {
        let val = self.0.borrow().value;
        let t = ((2.0 * val).exp() - 1.0) / ((2.0 * val).exp() + 1.0);
        let mut new_node = Scalar::new(t);
        new_node.set_op(Op::Tanh);
        let ins = Input::Unary(self.clone());
        new_node.set_ins(ins);

        new_node
    }

    pub fn exp(&self) -> Self {
        let mut new_node = Scalar::new(self.val().exp());
        new_node.set_op(Op::Exp);
        let ins = Input::Unary(self.clone());
        new_node.set_ins(ins);

        new_node
    }

    pub fn pow(&self, x: &Self) -> Self {
        let mut new_node = Scalar::new(self.val().powf(x.val()));
        new_node.set_op(Op::Pow);
        let ins = Input::Binary(self.clone(), x.clone());
        new_node.set_ins(ins);

        new_node
    }

    pub fn backward(&mut self) {
        self.set_grad(1.0);

        let sorted = self.topo_sort();
        let mut it = sorted.into_iter();
        while let Some(node) = it.next() {
            let node_grad = node.grad();
            match node.op() {
                Op::None => (),
                Op::Add => {
                    if let Some(Input::Binary(mut x, mut y)) = node.ins() {
                        x.add_grad(node_grad);
                        y.add_grad(node_grad);
                    }
                }
                Op::Mult => {
                    if let Some(Input::Binary(mut x, mut y)) = node.ins() {
                        let x_val = x.val();
                        let y_val = y.val();

                        x.add_grad(y_val * node_grad);
                        y.add_grad(x_val * node_grad);
                    }
                }
                Op::Exp => {
                    if let Some(Input::Unary(mut x)) = node.ins() {
                        x.add_grad(node_grad * node.val());
                    }
                }
                Op::Pow => {
                    if let Some(Input::Binary(mut base, exp)) = node.ins() {
                        let chained = node_grad * exp.val() * (base.val().powf(exp.val() - 1.0));
                        base.add_grad(chained);
                    }
                }
                Op::Tanh => {
                    if let Some(Input::Unary(mut x)) = node.ins() {
                        let x_val = x.val();
                        let grad = (1.0 - x_val.powi(2)) * node_grad;
                        x.add_grad(grad);
                    }
                }
            }
        }
    }

    fn topo_sort(&self) -> Vec<Self> {
        let mut result = Vec::new();
        let mut stack = VecDeque::new();
        let mut visited = HashSet::new();
        stack.push_back((self.clone(), false));

        while let Some(val) = stack.pop_back() {
            let (node, expanded) = val;
            let id = node.uuid().clone();
            if !visited.contains(&id) {
                let ins = node.ins();
                if !expanded {
                    stack.push_back((node, true));
                    if let Some(ins) = ins {
                        match ins {
                            Input::Unary(x) => stack.push_back((x, false)),
                            Input::Binary(x, y) => {
                                stack.push_back((x, false));
                                stack.push_back((y, false));
                            }
                        }
                    }
                } else {
                    visited.insert(id);
                    result.push(node);
                }
            }
        }
        result.reverse();

        result
    }
}

impl Add for Scalar {
    type Output = Scalar;

    fn add(self, rhs: Self) -> Self::Output {
        let result = self.0.borrow().value + rhs.0.borrow().value;
        let mut new_node = Scalar::new(result);
        new_node.set_op(Op::Add);
        let ins = Input::Binary(self.clone(), rhs.clone());
        new_node.set_ins(ins);

        new_node
    }
}

impl Sub for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        self * Scalar::new(-1.0)
    }
}

impl Mul for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Self::Output {
        let result = self.0.borrow().value * rhs.0.borrow().value;
        let mut new_node = Scalar::new(result);

        new_node.set_op(Op::Mult);
        let ins = Input::Binary(self.clone(), rhs.clone());
        new_node.set_ins(ins);

        new_node
    }
}

impl Div for Scalar {
    type Output = Scalar;

    fn div(self, rhs: Self) -> Self::Output {
        let rhs_inverse = rhs.pow(&Scalar::new(-1.0));
        self * rhs_inverse
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn addition() {
        let a = Scalar::new(1.0);
        let b = Scalar::new(2.0);
        let a_plus_b = a + b;
        assert_eq!(3.0, a_plus_b.val())
    }

    #[test]
    fn mul() {
        let a = Scalar::new(3.0);
        let b = Scalar::new(11.0);
        let a_times_b = a * b;
        assert_eq!(33.0, a_times_b.val())
    }

    #[test]
    fn exp() {
        let a = Scalar::new(3.0);
        let b = a.exp();
        assert_eq!((3.0 as f64).exp(), b.val())
    }

    #[test]
    fn tanh() {
        let x = Scalar::new(1.0);
        assert_eq!(0.7615941559557649, x.tanh().val())
    }

    #[test]
    fn backward() {
        let a = Scalar::new(5.0);
        let a_clone = a.clone();

        let b = Scalar::new(10.0);
        let b_clone = b.clone();

        let d = Scalar::new(1.0);
        let d_clone = d.clone();

        let c = a * b;
        let c_clone = c.clone();

        let mut e = c + d;

        e.backward();

        // gradients: e = 1.0
        //            d = 1.0
        //            c = 1.0
        //            b = 5.0
        //            a = 10.0

        assert_eq!(1.0, e.grad());
        assert_eq!(1.0, c_clone.grad());
        assert_eq!(1.0, d_clone.grad());
        assert_eq!(5.0, b_clone.grad());
        assert_eq!(10.0, a_clone.grad());
    }

    #[test]
    fn backward_multi_use() {
        let a = Scalar::new(5.0);
        let a_clone = a.clone();
        let mut b = a.clone() + a;

        b.backward();

        assert_eq!(2.0, a_clone.grad());
    }

    #[test]
    fn backward_neuron() {
        let x1 = Scalar::new(2.0);
        let x2 = Scalar::new(0.0);

        let w1 = Scalar::new(-3.0);
        let w2 = Scalar::new(1.0);

        let b = Scalar::new(6.8813735870195432);

        let x1w1 = x1.clone() * w1.clone();
        let _x1w2 = x1.clone() * w2.clone();

        let _x2w1 = x2.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();

        let x1w1_x2w2 = x1w1.clone() + x2w2.clone();

        let n = x1w1_x2w2.clone() + b.clone();

        let e = (Scalar::new(2.0) * n.clone()).exp();
        let mut o = (e.clone() - Scalar::new(1.0)) / (e + Scalar::new(1.0));

        o.backward();

        // o
        assert_eq!(0.7071067811865477, o.val());
        assert_eq!(1.0, o.grad());

        // n
        assert_eq!(0.8813735870195432, n.val());
        assert_eq!(0.5, n.grad());

        // x1w1 + x2w2
        assert_eq!(-6.0, x1w1_x2w2.val());
        assert_eq!(0.5, x1w1_x2w2.grad());

        // b
        assert_eq!(0.5, b.grad());

        // x2w2
        assert_eq!(0.0, x2w2.val());
        assert_eq!(0.5, x2w2.grad());

        // x1w1
        assert_eq!(-6.0, x1w1.val());
        assert_eq!(0.5, x1w1.grad());

        // x1
        assert_eq!(-1.5, x1.grad());
        // x2
        assert_eq!(0.5, x2.grad());
        // w1
        assert_eq!(1.0, w1.grad());
        // w2
        assert_eq!(0.0, w2.grad());
    }
}
