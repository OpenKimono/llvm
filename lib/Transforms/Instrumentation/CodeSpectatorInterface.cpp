#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h" // for itostr function
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

// see
// http://llvm.org/docs/ProgrammersManual.html#the-debug-macro-and-debug-option
// on how to use debugging infrastructure in LLVM
// also used by STATISTIC macro, so need to define this before using STATISTIC
#define DEBUG_TYPE "csi-func"

// XXX: Not sure how to turn these on yet
STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");

static const char *const CsiRtUnitInitName = "__csirt_unit_init";
static const char *const CsiRtUnitCtorName = "csirt.unit_ctor";
static const char *const CsiUnitBaseIdName = "__csi_unit_base_id";
static const char *const CsiUnitFedTableName = "__csi_unit_fed_table";
static const char *const CsiFuncIdVariablePrefix = "__csi_func_id_";

static const uint64_t CsiCallsiteUnknownTargetId = 0xffffffffffffffff;
// See llvm/tools/clang/lib/CodeGen/CodeGenModule.h:
static const int CsiUnitCtorPriority = 65535;

typedef struct {
    int32_t line;
    StringRef file;
} fed_entry_t;

namespace {

typedef struct {
  unsigned unused;
  bool unused2, unused3;
  bool read_before_write_in_bb;
} csi_acc_prop_t;

struct CodeSpectatorInterface : public ModulePass {
  static char ID;

  CodeSpectatorInterface() : ModulePass(ID) {}
  const char *getPassName() const override;
  bool doInitialization(Module &M) override;
  bool runOnModule(Module &M) override;
  bool runOnFunction(Function &F);
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  // not overriding doFinalization

private:
  int getNumBytesAccessed(Value *Addr, const DataLayout &DL);
  // initialize CSI instrumentation functions for load and store
  void initializeLoadStoreCallbacks(Module &M);
  // initialize CSI instrumentation functions for function entry and exit
  void initializeFuncCallbacks(Module &M);
  // Basic block entry and exit instrumentation
  void initializeBasicBlockCallbacks(Module &M);
  void initializeCallsiteCallbacks(Module &M);
  // actually insert the instrumentation call
  bool instrumentLoadOrStore(BasicBlock::iterator Iter, csi_acc_prop_t prop, const DataLayout &DL);

  void computeAttributesForMemoryAccesses(
      SmallVectorImpl<std::pair<BasicBlock::iterator, csi_acc_prop_t> > &Accesses,
      SmallVectorImpl<BasicBlock::iterator> &LocalAccesses);

  bool addLoadStoreInstrumentation(BasicBlock::iterator Iter,
                                   Function *BeforeFn,
                                   Function *AfterFn,
                                   Value *CsiId,
                                   Type *AddrType,
                                   Value *Addr,
                                   int NumBytes,
                                   csi_acc_prop_t prop);
  // instrument a call to memmove, memcpy, or memset
  void instrumentMemIntrinsic(BasicBlock::iterator I);
  void instrumentCallsite(CallSite &CS);
  bool instrumentBasicBlock(BasicBlock &BB);
  bool FunctionCallsFunction(Function *F, Function *G);
  bool ShouldNotInstrumentFunction(Function &F);
  void InitializeCsi(Module &M);
  void FinalizeCsi(Module &M);
  Value *InsertFedTable(Module &M);

  SmallVector<Constant *, 4> ConvertFEDEntriesToConsts(Module &M);

  // Inserts computation of the CSI id for the given instrumentation call,
  // assuming that the call corresponds to the most recent entry in FedEntries.
  // Returns the csi id as a value, which can be passed as an argument to the
  // call.
  Value *InsertCsiIdComputation(IRBuilder<> IRB);

  CallGraph *CG;

  GlobalVariable *UnitBaseId;

  Function *CsiBeforeRead;
  Function *CsiAfterRead;
  Function *CsiBeforeWrite;
  Function *CsiAfterWrite;

  Function *CsiFuncEntry;
  Function *CsiFuncExit;
  Function *CsiBBEntry, *CsiBBExit;
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *CsiBeforeCallsite;

  Type *IntptrTy;

  SmallVector<fed_entry_t, 4> FedEntries;

  std::map<std::string, uint64_t> FuncOffsetMap;
}; //struct CodeSpectatorInterface
} //namespace

// the address matters but not the init value
char CodeSpectatorInterface::ID = 0;
INITIALIZE_PASS(CodeSpectatorInterface, "CSI-func", "CodeSpectatorInterface function pass",
                false, false)

const char *CodeSpectatorInterface::getPassName() const {
  return "CodeSpectatorInterface";
}

ModulePass *llvm::createCodeSpectatorInterfacePass() {
  return new CodeSpectatorInterface();
}

/**
 * initialize the declaration of function call instrumentation functions
 *
 * void __csi_func_entry(uint64_t csi_id, void *function, void *return_addr, char *func_name);
 * void __csi_func_exit(uint64_t csi_id, void *function, void *return_addr, char *func_name);
 */
void CodeSpectatorInterface::initializeFuncCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  CsiFuncEntry = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_entry", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), nullptr));
  CsiFuncExit = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_func_exit", IRB.getVoidTy(), IRB.getInt64Ty(), IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), nullptr));
}

void CodeSpectatorInterface::initializeBasicBlockCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  SmallVector<Type *, 4> ArgTypes({IRB.getInt64Ty()});
  FunctionType *FnType = FunctionType::get(IRB.getVoidTy(), ArgTypes, false);
  CsiBBEntry = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_bb_entry", FnType));

  CsiBBExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_bb_exit", FnType));
}

void CodeSpectatorInterface::initializeCallsiteCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  SmallVector<Type *, 4> ArgTypes({IRB.getInt64Ty(), IRB.getInt64Ty()});
  FunctionType *FnType = FunctionType::get(IRB.getVoidTy(), ArgTypes, false);
  CsiBeforeCallsite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_callsite", FnType));
}

/**
 * initialize the declaration of instrumentation functions
 *
 * void __csi_before_load(uint64_t csi_id, void *addr, uint32_t num_bytes, uint64_t prop);
 *
 * where num_bytes = 1, 2, 4, 8.
 *
 * Presumably aligned / unaligned accesses are specified by the attr
 */
void CodeSpectatorInterface::initializeLoadStoreCallbacks(Module &M) {

  IRBuilder<> IRB(M.getContext());
  Type *RetType = IRB.getVoidTy();            // return void
  Type *AddrType = IRB.getInt8PtrTy();        // void *addr
  Type *NumBytesType = IRB.getInt32Ty();      // int num_bytes

  // Initialize the instrumentation for reads, writes

  // void __csi_before_load(uint64_t csi_id, void *addr, int num_bytes, int attr);
  CsiBeforeRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_load", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  // void __csi_after_load(uint64_t csi_id, void *addr, int num_bytes, int attr);
  SmallString<32> AfterReadName("__csi_after_load");
  CsiAfterRead = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_load", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  // void __csi_before_store(uint64_t csi_id, void *addr, int num_bytes, int attr);
  CsiBeforeWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_store", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  // void __csi_after_store(uint64_t csi_id, void *addr, int num_bytes, int attr);
  CsiAfterWrite = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_store", RetType,
        IRB.getInt64Ty(), AddrType, NumBytesType, IRB.getInt64Ty(), nullptr));

  MemmoveFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemcpyFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemsetFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt32Ty(), IntptrTy, nullptr));
}

int CodeSpectatorInterface::getNumBytesAccessed(Value *Addr,
                                                const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    DEBUG_WITH_TYPE("csi-func",
        errs() << "Bad size " << TypeSize << " at addr " << Addr << "\n");
    NumAccessesWithBadSize++;
    return -1;
  }
  return TypeSize / 8;
}

bool CodeSpectatorInterface::addLoadStoreInstrumentation(BasicBlock::iterator Iter,
                                                         Function *BeforeFn,
                                                         Function *AfterFn,
                                                         Value *CsiId,
                                                         Type *AddrType,
                                                         Value *Addr,
                                                         int NumBytes,
                                                         csi_acc_prop_t prop) {
  IRBuilder<> IRB(&(*Iter));
  IRB.CreateCall(BeforeFn,
      // XXX: should I just use the pointer type with the right size?
      {CsiId,
       IRB.CreatePointerCast(Addr, AddrType),
       IRB.getInt32(NumBytes),
       IRB.getInt64(0)});  // TODO(ddoucet): fix this
       /* IRB.getInt32(prop.unused),
       IRB.getInt1(prop.unused2),
       IRB.getInt1(prop.unused3),
       IRB.getInt1(prop.read_before_write_in_bb)}); */

  // The iterator currently points between the inserted instruction and the
  // store instruction. We now want to insert an instruction after the store
  // instruction.
  Iter++;
  IRB.SetInsertPoint(&*Iter);

  IRB.CreateCall(AfterFn,
      {CsiId,
       IRB.CreatePointerCast(Addr, AddrType),
       IRB.getInt32(NumBytes),
       IRB.getInt64(0)});  // TODO(ddoucet): fix this
       /* IRB.getInt32(prop.unused),
       IRB.getInt1(prop.unused2),
       IRB.getInt1(prop.unused3),
       IRB.getInt1(prop.read_before_write_in_bb)}); */

  return true;
}

bool CodeSpectatorInterface::instrumentLoadOrStore(BasicBlock::iterator Iter,
                                                   csi_acc_prop_t prop,
                                                   const DataLayout &DL) {

  DEBUG_WITH_TYPE("csi-func",
      errs() << "CSI_func: instrument instruction " << *Iter << "\n");

  Instruction *I = &(*Iter);
  // takes pointer to Instruction and inserts before the instruction
  IRBuilder<> IRB(&(*Iter));
  bool IsWrite = isa<StoreInst>(I);
  Value *Addr = IsWrite ?
      cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();

  int NumBytes = getNumBytesAccessed(Addr, DL);
  Type *AddrType = IRB.getInt8PtrTy();

  if (NumBytes == -1) return false; // size that we don't recognize

  bool Res = false;

  if (DILocation *Loc = I->getDebugLoc()) {
    FedEntries.push_back(fed_entry_t{(int32_t)Loc->getLine(), Loc->getFilename()});
  } else {
    // Not much we can do here
    FedEntries.push_back(fed_entry_t{-1, ""});
  }
  Value *CsiId = InsertCsiIdComputation(IRB);

  if(IsWrite) {
    Res = addLoadStoreInstrumentation(
        Iter, CsiBeforeWrite, CsiAfterWrite, CsiId, AddrType, Addr, NumBytes, prop);
    NumInstrumentedWrites++;

  } else { // is read
    Res = addLoadStoreInstrumentation(
        Iter, CsiBeforeRead, CsiAfterRead, CsiId, AddrType, Addr, NumBytes, prop);
    NumInstrumentedReads++;
  }

  return Res;
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since our pass runs after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __csi_memset to avoid the intrinsics.
void CodeSpectatorInterface::instrumentMemIntrinsic(BasicBlock::iterator Iter) {
  Instruction *I = &(*Iter);
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    IRB.CreateCall(
        MemsetFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    IRB.CreateCall(
        isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(M->getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  }
}

DILocation *getFirstDebugLoc(BasicBlock &BB) {
  for (Instruction &Inst : BB)
    if (DILocation *Loc = Inst.getDebugLoc())
      return Loc;

  return nullptr;
}

// Assumes that the instrumentation has already been inserted into the FedEntries vector.
Value *CodeSpectatorInterface::InsertCsiIdComputation(IRBuilder<> IRB) {
  Value *unitBaseId = IRB.CreateLoad(UnitBaseId);
  Value *localOffset = IRB.getInt64((uint64_t)FedEntries.size() - 1);
  return IRB.CreateAdd(unitBaseId, localOffset);
}

bool CodeSpectatorInterface::instrumentBasicBlock(BasicBlock &BB) {
  if (DILocation *Loc = getFirstDebugLoc(BB)) {
    FedEntries.push_back(fed_entry_t{(int32_t)Loc->getLine(), Loc->getFilename()});
  } else {
    // Not much we can do here
    FedEntries.push_back(fed_entry_t{-1, ""});
  }

  IRBuilder<> IRB(BB.getFirstInsertionPt());
  Value *CsiId = InsertCsiIdComputation(IRB);

  IRB.CreateCall(CsiBBEntry, {CsiId});

  TerminatorInst *TI = BB.getTerminator();
  IRB.SetInsertPoint(TI);
  IRB.CreateCall(CsiBBExit, {CsiId});
  return true;
}

void CodeSpectatorInterface::instrumentCallsite(CallSite &CS) {
  Instruction *I = CS.getInstruction();
  Module *M = I->getParent()->getParent()->getParent();
  Function *Called = CS.getCalledFunction();

  if (Called && Called->getName().startswith("llvm.dbg")) {
      return;
  }

  if (DILocation *Loc = I->getDebugLoc()) {
    FedEntries.push_back(fed_entry_t{(int32_t)Loc->getLine(), Loc->getFilename()});
  } else {
    FedEntries.push_back(fed_entry_t{-1, ""});
  }

  IRBuilder<> IRB(I);
  Value *CsiId = InsertCsiIdComputation(IRB);

  std::string GVName = CsiFuncIdVariablePrefix + Called->getName().str();
  GlobalVariable *FuncIdGV = dyn_cast<GlobalVariable>(M->getOrInsertGlobal(GVName, IRB.getInt64Ty()));
  assert(FuncIdGV);
  FuncIdGV->setConstant(false);
  FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
  FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));

  Value *FuncId = IRB.CreateLoad(FuncIdGV);
  IRB.CreateCall(CsiBeforeCallsite, {CsiId, FuncId});
}

bool CodeSpectatorInterface::doInitialization(Module &M) {
  DEBUG_WITH_TYPE("csi-func", errs() << "CSI_func: doInitialization" << "\n");

  IntptrTy = M.getDataLayout().getIntPtrType(M.getContext());

  DEBUG_WITH_TYPE("csi-func",
      errs() << "CSI_func: doInitialization done" << "\n");
  return true;
}

void CodeSpectatorInterface::InitializeCsi(Module &M) {
  LLVMContext &C = M.getContext();
  IntegerType *Int64Ty = IntegerType::get(C, 64);

  UnitBaseId = new GlobalVariable(M, Int64Ty, false, GlobalValue::InternalLinkage, ConstantInt::get(Int64Ty, 0), CsiUnitBaseIdName);
  assert(UnitBaseId);

  initializeFuncCallbacks(M);
  initializeLoadStoreCallbacks(M);
  initializeBasicBlockCallbacks(M);
  initializeCallsiteCallbacks(M);

  CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
}

StructType *CreateFedEntryType(LLVMContext &C) {
  return StructType::get(
      IntegerType::get(C, 32),
      PointerType::get(IntegerType::get(C, 8), 0),
      nullptr);
}

SmallVector<Constant *, 4> CodeSpectatorInterface::ConvertFEDEntriesToConsts(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *FedType = CreateFedEntryType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);

  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};

  IRBuilder<> IRB(C);
  SmallVector<Constant *, 4> Ret;

  for (fed_entry_t &Entry : FedEntries) {
    Value *Line = ConstantInt::get(Int32Ty, Entry.line);

    // TODO(ddoucet): It'd be nice to reuse the global variables since most
    // module names will be the same. Do the pointers have the same value as well
    // or do we actually have to hash the string?
    Constant *FileStrConstant = ConstantDataArray::getString(C, Entry.file);
    GlobalVariable *GV = new GlobalVariable(M, FileStrConstant->getType(),
                                            true, GlobalValue::PrivateLinkage,
                                            FileStrConstant, "", nullptr,
                                            GlobalVariable::NotThreadLocal, 0);
    GV->setUnnamedAddr(true);
    Constant *File = ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);

    Ret.push_back(ConstantStruct::get(FedType, Line, File, nullptr));
  }
  return Ret;
}

// Returns a pointer to the first element
Value *CodeSpectatorInterface::InsertFedTable(Module &M) {
  LLVMContext &C = M.getContext();

  SmallVector<Constant *, 4> FedEntries = ConvertFEDEntriesToConsts(M);
  ArrayType *FedArrayType = ArrayType::get(CreateFedEntryType(C), FedEntries.size());

  Constant *Table = ConstantArray::get(FedArrayType, FedEntries);
  GlobalVariable *GV = new GlobalVariable(
      M, FedArrayType, false, GlobalValue::InternalLinkage, Table, CsiUnitFedTableName);

  Constant *Zero = ConstantInt::get(IntegerType::get(C, 32), 0);
  Value *GepArgs[] = {Zero, Zero};
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

void CodeSpectatorInterface::FinalizeCsi(Module &M) {
  LLVMContext &C = M.getContext();

  // Add CSI global constructor, which calls unit init.
  Function *Ctor = Function::Create(
      FunctionType::get(Type::getVoidTy(C), false),
      GlobalValue::InternalLinkage, CsiRtUnitCtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({
      IRB.getInt8PtrTy(),
      IRB.getInt64Ty(),
      PointerType::get(IRB.getInt64Ty(), 0),
      PointerType::get(CreateFedEntryType(C), 0)
  });
  FunctionType *InitFunctionTy = FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  Function *InitFunction = checkCsiInterfaceFunction(
      M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy));
  assert(InitFunction);

  // Insert __csi_func_id_<f> weak symbols for all defined functions
  // and generate the runtime code that stores to all of them.
  LoadInst *LI = IRB.CreateLoad(UnitBaseId);
  for (const auto &it : FuncOffsetMap) {
    std::string GVName = CsiFuncIdVariablePrefix + it.first;
    GlobalVariable *GV = nullptr;
    if ((GV = M.getGlobalVariable(GVName)) == nullptr) {
        GV = new GlobalVariable(M, IRB.getInt64Ty(), false, GlobalValue::WeakAnyLinkage, IRB.getInt64(CsiCallsiteUnknownTargetId), GVName);
    }
    assert(GV);
    IRB.CreateStore(IRB.CreateAdd(LI, IRB.getInt64(it.second)), GV);
  }

  // Insert call to __csirt_unit_init
  CallInst *Call = IRB.CreateCall(InitFunction, {
      IRB.CreateGlobalStringPtr(M.getName()),
      IRB.getInt64((int64_t)FedEntries.size()),
      UnitBaseId,
      InsertFedTable(M)
  });

  // Add the constructor to the global list
  appendToGlobalCtors(M, Ctor, CsiUnitCtorPriority);

  CallGraphNode *CNCtor = CG->getOrInsertFunction(Ctor);
  CallGraphNode *CNFunc = CG->getOrInsertFunction(InitFunction);
  CNCtor->addCalledFunction(Call, CNFunc);
}

void CodeSpectatorInterface::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
}

// Recursively determine if F calls G. Return true if so. Conservatively, if F makes
// any internal indirect function calls, assume it calls G.
bool CodeSpectatorInterface::FunctionCallsFunction(Function *F, Function *G) {
  assert(F && G && CG);
  CallGraphNode *CGN = (*CG)[F];
  // Assume external functions cannot make calls to internal functions.
  if (!F->hasLocalLinkage() && G->hasLocalLinkage()) return false;
  // Assume function declarations won't make calls to internal
  // functions. TODO: This may not be correct in general.
  if (F->isDeclaration()) return false;
  for (CallGraphNode::iterator it = CGN->begin(), ite = CGN->end(); it != ite; ++it) {
    Function *Called = it->second->getFunction();
    if (Called == NULL) {
      // Indirect call
      return true;
    } else if (Called == G) {
      return true;
    } else if (G->hasLocalLinkage() && !Called->hasLocalLinkage()) {
      // Assume external functions cannot make calls to internal functions.
      continue;
    }
  }
  for (CallGraphNode::iterator it = CGN->begin(), ite = CGN->end(); it != ite; ++it) {
    Function *Called = it->second->getFunction();
    if (FunctionCallsFunction(Called, G)) return true;
  }
  return false;
}

bool CodeSpectatorInterface::ShouldNotInstrumentFunction(Function &F) {
    Module &M = *F.getParent();
    if (F.hasName() && F.getName() == CsiRtUnitCtorName) {
        return true;
    }
    // Don't instrument functions that will run before or
    // simultaneously with CSI ctors.
    GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors");
    if (GV == nullptr) return false;
    ConstantArray *CA = cast<ConstantArray>(GV->getInitializer());
    for (Use &OP : CA->operands()) {
        if (isa<ConstantAggregateZero>(OP)) continue;
        ConstantStruct *CS = cast<ConstantStruct>(OP);

        if (Function *CF = dyn_cast<Function>(CS->getOperand(1))) {
            uint64_t Priority = dyn_cast<ConstantInt>(CS->getOperand(0))->getLimitedValue();
            if (Priority <= CsiUnitCtorPriority) {
                return CF->getName() == F.getName() ||  FunctionCallsFunction(CF, &F);
            }
        }
    }
    // false means do instrument it.
    return false;
}

void CodeSpectatorInterface::computeAttributesForMemoryAccesses(
    SmallVectorImpl<std::pair<BasicBlock::iterator, csi_acc_prop_t> > &MemoryAccesses,
    SmallVectorImpl<BasicBlock::iterator> &LocalAccesses) {
  SmallSet<Value*, 8> WriteTargets;

  for (SmallVectorImpl<BasicBlock::iterator>::reverse_iterator It = LocalAccesses.rbegin(),
      E = LocalAccesses.rend(); It != E; ++It) {
    BasicBlock::iterator II = *It;
    Instruction *I = &(*II);
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      WriteTargets.insert(Store->getPointerOperand());
      MemoryAccesses.push_back(
        std::make_pair(II, csi_acc_prop_t{0, false, false, false}));
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      bool HasBeenSeen = WriteTargets.count(Addr) > 0;
      MemoryAccesses.push_back(
        std::make_pair(II, csi_acc_prop_t{0, false, false, HasBeenSeen}));
    }
  }
  LocalAccesses.clear();
}

bool CodeSpectatorInterface::runOnModule(Module &M) {
  InitializeCsi(M);

  for (Function &F : M)
    runOnFunction(F);

  FinalizeCsi(M);
  return true;  // we always insert the unit constructor
}

bool CodeSpectatorInterface::runOnFunction(Function &F) {
  // This is required to prevent instrumenting the call to
  // __csi_module_init from within the module constructor.
  if (F.empty() || ShouldNotInstrumentFunction(F)) {
      return false;
  }

  DEBUG_WITH_TYPE("csi-func",
                  errs() << "CSI_func: run on function " << F.getName() << "\n");

  SmallVector<std::pair<BasicBlock::iterator, csi_acc_prop_t>, 8> MemoryAccesses;
  SmallSet<Value*, 8> WriteTargets;
  SmallVector<BasicBlock::iterator, 8> LocalMemoryAccesses;

  SmallVector<BasicBlock::iterator, 8> RetVec;
  SmallVector<BasicBlock::iterator, 8> MemIntrinsics;
  SmallVector<BasicBlock::iterator, 8> Callsites;
  bool Modified = false;
  const DataLayout &DL = F.getParent()->getDataLayout();

  FuncOffsetMap[F.getName()] = FuncOffsetMap.size();

  // Traverse all instructions in a function and insert instrumentation
  // on load & store
  for (BasicBlock &BB : F) {
    for (auto II = BB.begin(); II != BB.end(); II++) {
      Instruction *I = &(*II);
      if (isa<LoadInst>(*I) || isa<StoreInst>(*I)) {
        LocalMemoryAccesses.push_back(II);
      } else if (isa<ReturnInst>(*I)) {
        RetVec.push_back(II);
      } else if (isa<CallInst>(*I) || isa<InvokeInst>(*I)) {
        Callsites.push_back(II);
        if (isa<MemIntrinsic>(I))
          MemIntrinsics.push_back(II);
        computeAttributesForMemoryAccesses(MemoryAccesses, LocalMemoryAccesses);
      }
    }
    computeAttributesForMemoryAccesses(MemoryAccesses, LocalMemoryAccesses);
  }

  // Do this work in a separate loop after copying the iterators so that we
  // aren't modifying the list as we're iterating.
  for (std::pair<BasicBlock::iterator, csi_acc_prop_t> p : MemoryAccesses)
    Modified |= instrumentLoadOrStore(p.first, p.second, DL);

  for (BasicBlock::iterator I : MemIntrinsics)
    instrumentMemIntrinsic(I);

  for (BasicBlock::iterator I : Callsites) {
    CallSite CS(I);
    instrumentCallsite(CS);
  }

  // Instrument basic blocks
  // Note that we do this before function entry so that we put this at the
  // beginning of the basic block, and then the function entry call goes before
  // the call to basic block entry.
  for (BasicBlock &BB : F) {
    Modified |= instrumentBasicBlock(BB);
  }

  // Instrument function entry/exit points.
  IRBuilder<> IRB(F.getEntryBlock().getFirstInsertionPt());

  if (DISubprogram *Subprog = llvm::getDISubprogram(&F)) {
    FedEntries.push_back(fed_entry_t{(int32_t)Subprog->getLine(), Subprog->getFilename()});
  } else {
    FedEntries.push_back(fed_entry_t{-1, ""});
  }
  Value *CsiId = InsertCsiIdComputation(IRB);

  Value *Function = ConstantExpr::getBitCast(&F, IRB.getInt8PtrTy());
  Value *FunctionName = IRB.CreateGlobalStringPtr(F.getName());
  Value *ReturnAddress = IRB.CreateCall(
      Intrinsic::getDeclaration(F.getParent(), Intrinsic::returnaddress),
      IRB.getInt32(0));
  IRB.CreateCall(CsiFuncEntry, {CsiId, Function, ReturnAddress, FunctionName});

  for (BasicBlock::iterator I : RetVec) {
      Instruction *RetInst = &(*I);
      IRBuilder<> IRBRet(RetInst);
      IRBRet.CreateCall(CsiFuncExit, {CsiId, Function, ReturnAddress, FunctionName});
  }
  Modified = true;

  if(Modified) {
    DEBUG_WITH_TYPE("csi-func",
        errs() << "CSI_func: modified function " << F.getName() << "\n");
  }
  return Modified;
}

// End of compile-time pass
// ------------------------------------------------------------------------
