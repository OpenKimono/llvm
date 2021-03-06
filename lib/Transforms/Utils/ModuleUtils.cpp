//===-- ModuleUtils.cpp - Functions to manipulate Modules -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on Modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void appendToGlobalArray(const char *Array, Module &M, Function *F,
                                int Priority, Constant *Data) {
  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);

  // Get the current set of static global constructors and add the new ctor
  // to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  StructType *EltTy;
  if (GlobalVariable *GVCtor = M.getNamedGlobal(Array)) {
    ArrayType *ATy = cast<ArrayType>(GVCtor->getValueType());
    StructType *OldEltTy = cast<StructType>(ATy->getElementType());
    // Upgrade a 2-field global array type to the new 3-field format if needed.
    if (Data && OldEltTy->getNumElements() < 3)
      EltTy = StructType::get(IRB.getInt32Ty(), PointerType::getUnqual(FnTy),
                              IRB.getInt8PtrTy(), nullptr);
    else
      EltTy = OldEltTy;
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned n = Init->getNumOperands();
      CurrentCtors.reserve(n + 1);
      for (unsigned i = 0; i != n; ++i) {
        auto Ctor = cast<Constant>(Init->getOperand(i));
        if (EltTy != OldEltTy)
          Ctor = ConstantStruct::get(
              EltTy, Ctor->getAggregateElement((unsigned)0),
              Ctor->getAggregateElement(1),
              Constant::getNullValue(IRB.getInt8PtrTy()), nullptr);
        CurrentCtors.push_back(Ctor);
      }
    }
    GVCtor->eraseFromParent();
  } else {
    // Use the new three-field struct if there isn't one already.
    EltTy = StructType::get(IRB.getInt32Ty(), PointerType::getUnqual(FnTy),
                            IRB.getInt8PtrTy(), nullptr);
  }

  // Build a 2 or 3 field global_ctor entry.  We don't take a comdat key.
  Constant *CSVals[3];
  CSVals[0] = IRB.getInt32(Priority);
  CSVals[1] = F;
  // FIXME: Drop support for the two element form in LLVM 4.0.
  if (EltTy->getNumElements() >= 3)
    CSVals[2] = Data ? ConstantExpr::getPointerCast(Data, IRB.getInt8PtrTy())
                     : Constant::getNullValue(IRB.getInt8PtrTy());
  Constant *RuntimeCtorInit =
      ConstantStruct::get(EltTy, makeArrayRef(CSVals, EltTy->getNumElements()));

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(EltTy, CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit, Array);
}

void llvm::appendToGlobalCtors(Module &M, Function *F, int Priority, Constant *Data) {
  appendToGlobalArray("llvm.global_ctors", M, F, Priority, Data);
}

void llvm::appendToGlobalDtors(Module &M, Function *F, int Priority, Constant *Data) {
  appendToGlobalArray("llvm.global_dtors", M, F, Priority, Data);
}

// XXX: Should really combine the two functions below, but didn't want to
// change all files that uses checkSanitizerInterfaceFunction, so leave them
// alone for now ...
Function *llvm::checkCsiInterfaceFunction(Constant *FuncOrBitcast) {
  if (Function *F = dyn_cast<Function>(FuncOrBitcast)) {
    return F;
  }
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FuncOrBitcast)) {
    if (CE->isCast() && CE->getOpcode() == Instruction::BitCast) {
      if (Function *F = dyn_cast<Function>(CE->getOperand(0))) {
        return F;
      }
    }
  }
  FuncOrBitcast->dump();
  std::string Err;
  raw_string_ostream Stream(Err);
  Stream << "CodeSpectatorInterface interface function redefined: " << *FuncOrBitcast;
  report_fatal_error(Err);
}

Function *llvm::checkSanitizerInterfaceFunction(Constant *FuncOrBitcast) {
  if (isa<Function>(FuncOrBitcast))
    return cast<Function>(FuncOrBitcast);
  FuncOrBitcast->dump();
  std::string Err;
  raw_string_ostream Stream(Err);
  Stream << "Sanitizer interface function redefined: " << *FuncOrBitcast;
  report_fatal_error(Err);
}

std::pair<Function *, Function *> llvm::createSanitizerCtorAndInitFunctions(
    Module &M, StringRef CtorName, StringRef InitName,
    ArrayRef<Type *> InitArgTypes, ArrayRef<Value *> InitArgs,
    StringRef VersionCheckName) {
  assert(!InitName.empty() && "Expected init function name");
  assert(InitArgTypes.size() == InitArgTypes.size() &&
         "Sanitizer's init function expects different number of arguments");
  Function *Ctor = Function::Create(
      FunctionType::get(Type::getVoidTy(M.getContext()), false),
      GlobalValue::InternalLinkage, CtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(M.getContext(), "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(M.getContext(), CtorBB));
  Function *InitFunction =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          InitName, FunctionType::get(IRB.getVoidTy(), InitArgTypes, false),
          AttributeSet()));
  InitFunction->setLinkage(Function::ExternalLinkage);
  IRB.CreateCall(InitFunction, InitArgs);
  if (!VersionCheckName.empty()) {
    Function *VersionCheckFunction =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            VersionCheckName, FunctionType::get(IRB.getVoidTy(), {}, false),
            AttributeSet()));
    IRB.CreateCall(VersionCheckFunction, {});
  }
  return std::make_pair(Ctor, InitFunction);
}
