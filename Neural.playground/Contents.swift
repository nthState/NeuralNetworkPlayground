//: Playground - noun: a place where people can play

import Cocoa

let n = Network()

// Before Training
n.feedForward(inputValues: [0,0])
n.feedForward(inputValues: [1,0])

// Train
n.multitrain()

// After training
n.feedForward(inputValues: [0,0]) // expect 0
n.feedForward(inputValues: [0,1]) // expect 1
n.feedForward(inputValues: [1,0]) // expect 1
n.feedForward(inputValues: [1,1]) // expect 0

//n.feedForward(inputValues: [0.7,0]) // expect 1
//n.feedForward(inputValues: [0.3,0]) // expect 1

n.debugPrint()
